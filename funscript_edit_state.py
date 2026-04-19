"""Editable funscript state with selection, undo/redo, and locked regions."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from PyQt6.QtCore import QObject, pyqtSignal

from pmv_funscript_io import FunscriptAction


MAX_UNDO_STACK = 50


@dataclass
class LockedRegion:
    """A time range whose actions are preserved across regeneration."""
    start_ms: int
    end_ms: int


@dataclass
class EditSnapshot:
    """Immutable copy of the action list + locked regions for undo/redo."""
    actions: list[FunscriptAction]
    locked_regions: list[LockedRegion]
    selection: frozenset[int]
    description: str


class FunscriptEditState(QObject):
    """
    Maintains the editable funscript action list, selection,
    locked regions, clipboard, and undo/redo stacks.

    All mutation goes through this class so undo snapshots are consistent.
    """

    changed = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._actions: list[FunscriptAction] = []
        self._selection: set[int] = set()
        self._locked_regions: list[LockedRegion] = []
        self._clean_actions: list[FunscriptAction] = []
        self._clean_locked_regions: list[LockedRegion] = []
        self._clipboard: list[FunscriptAction] = []
        self._undo_stack: list[EditSnapshot] = []
        self._redo_stack: list[EditSnapshot] = []
        self._dirty: bool = False
        self._multi_axis_stale: bool = False
        self._version: int = 0

    # ── Properties ──────────────────────────────────────────

    @property
    def actions(self) -> list[FunscriptAction]:
        return self._actions

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def multi_axis_stale(self) -> bool:
        return self._multi_axis_stale

    @property
    def version(self) -> int:
        return self._version

    @property
    def selection_indices(self) -> set[int]:
        return self._selection

    @property
    def selected_actions(self) -> list[FunscriptAction]:
        return [self._actions[i] for i in sorted(self._selection) if i < len(self._actions)]

    @property
    def locked_regions(self) -> list[LockedRegion]:
        return self._locked_regions

    @property
    def has_selection(self) -> bool:
        return len(self._selection) > 0

    @property
    def clipboard_empty(self) -> bool:
        return len(self._clipboard) == 0

    def is_locked(self, time_ms: int) -> bool:
        """Return True if the given timestamp falls inside any locked region."""
        for r in self._locked_regions:
            if r.start_ms <= time_ms <= r.end_ms:
                return True
        return False

    def is_in_selection(self, index: int) -> bool:
        return index in self._selection

    # ── Internal helpers ────────────────────────────────────

    def _mark_dirty(self) -> None:
        self._set_state_flags(True)

    def _emit_changed(self) -> None:
        """Emit changed without marking dirty (e.g., selection-only changes)."""
        self.changed.emit()

    @staticmethod
    def _clone_actions(actions: list[FunscriptAction]) -> list[FunscriptAction]:
        return [FunscriptAction(a.at, a.pos) for a in actions]

    @staticmethod
    def _clone_locked_regions(regions: list[LockedRegion]) -> list[LockedRegion]:
        return [LockedRegion(r.start_ms, r.end_ms) for r in regions]

    @staticmethod
    def _actions_equal(left: list[FunscriptAction], right: list[FunscriptAction]) -> bool:
        return len(left) == len(right) and all(a.at == b.at and a.pos == b.pos for a, b in zip(left, right))

    @staticmethod
    def _locked_regions_equal(left: list[LockedRegion], right: list[LockedRegion]) -> bool:
        return len(left) == len(right) and all(
            a.start_ms == b.start_ms and a.end_ms == b.end_ms for a, b in zip(left, right)
        )

    def _make_snapshot(self, description: str) -> EditSnapshot:
        return EditSnapshot(
            actions=self._clone_actions(self._actions),
            locked_regions=self._clone_locked_regions(self._locked_regions),
            selection=frozenset(self._selection),
            description=description,
        )

    def _reset_clean_state(self) -> None:
        self._clean_actions = self._clone_actions(self._actions)
        self._clean_locked_regions = self._clone_locked_regions(self._locked_regions)

    def _set_state_flags(self, dirty: bool | None = None) -> None:
        if dirty is None:
            dirty = not (
                self._actions_equal(self._actions, self._clean_actions)
                and self._locked_regions_equal(self._locked_regions, self._clean_locked_regions)
            )
        self._dirty = bool(dirty)
        self._multi_axis_stale = bool(dirty)
        self._version += 1
        self.changed.emit()

    def _snapshot(self, description: str) -> None:
        """Push current state onto undo stack, clear redo stack."""
        self._undo_stack.append(self._make_snapshot(description))
        if len(self._undo_stack) > MAX_UNDO_STACK:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _sort_actions(self) -> None:
        """Re-sort actions by time and rebuild selection indices."""
        if not self._actions:
            self._selection.clear()
            return
        # Track selected actions by (at, pos) value pairs before sort
        selected_values = [
            (self._actions[i].at, self._actions[i].pos)
            for i in self._selection if i < len(self._actions)
        ]
        self._actions.sort(key=lambda a: a.at)
        # Rebuild selection by matching values (greedy, handles duplicates)
        remaining = list(selected_values)
        new_sel: set[int] = set()
        for i, a in enumerate(self._actions):
            key = (a.at, a.pos)
            if key in remaining:
                new_sel.add(i)
                remaining.remove(key)
        self._selection = new_sel

    # ── Bulk load (from generation or file import) ──────────

    def load_actions(self, actions: list[FunscriptAction]) -> None:
        """Replace all actions. Clears selection and undo history."""
        self._actions = self._clone_actions(actions)
        self._actions.sort(key=lambda a: a.at)
        self._selection.clear()
        self._locked_regions.clear()
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._reset_clean_state()
        self._set_state_flags(False)

    def accept_generation(self, actions: list[FunscriptAction]) -> None:
        """Accept fresh generation result — snapshots first so it's undoable."""
        self._snapshot("Generate")
        self._actions = self._clone_actions(actions)
        self._actions.sort(key=lambda a: a.at)
        self._selection.clear()
        self._reset_clean_state()
        self._set_state_flags(False)

    # ── Selection ───────────────────────────────────────────

    def select_index(self, idx: int, toggle: bool = False) -> None:
        if idx < 0 or idx >= len(self._actions):
            return
        if toggle:
            if idx in self._selection:
                self._selection.discard(idx)
            else:
                self._selection.add(idx)
        else:
            self._selection = {idx}
        self._emit_changed()

    def select_range(self, start_ms: int, end_ms: int) -> None:
        """Select all actions within the given time range."""
        lo = min(start_ms, end_ms)
        hi = max(start_ms, end_ms)
        for i, a in enumerate(self._actions):
            if lo <= a.at <= hi:
                self._selection.add(i)
        self._emit_changed()

    def select_rect(self, start_ms: int, end_ms: int, pos_lo: float, pos_hi: float) -> None:
        """Select all actions within a rectangle (time + position range)."""
        t_lo = min(start_ms, end_ms)
        t_hi = max(start_ms, end_ms)
        p_lo = min(pos_lo, pos_hi)
        p_hi = max(pos_lo, pos_hi)
        for i, a in enumerate(self._actions):
            if t_lo <= a.at <= t_hi and p_lo <= a.pos <= p_hi:
                self._selection.add(i)
        self._emit_changed()

    def select_all(self) -> None:
        self._selection = set(range(len(self._actions)))
        self._emit_changed()

    def clear_selection(self) -> None:
        self._selection.clear()
        self._emit_changed()

    def select_top_points(self) -> None:
        """Select all actions at position >= 90."""
        for i, a in enumerate(self._actions):
            if a.pos >= 90:
                self._selection.add(i)
        self._emit_changed()

    def select_bottom_points(self) -> None:
        """Select all actions at position <= 10."""
        for i, a in enumerate(self._actions):
            if a.pos <= 10:
                self._selection.add(i)
        self._emit_changed()

    # ── Point editing (all snapshot before mutating) ────────

    def add_action(self, action: FunscriptAction) -> None:
        if self.is_locked(action.at):
            return
        self._snapshot("Add point")
        new_at = action.at
        new_pos = max(0, min(100, action.pos))
        new_action = FunscriptAction(new_at, new_pos)
        bisect.insort(self._actions, new_action, key=lambda a: a.at)
        idx = self._find_action_index(new_at, new_pos)
        self._selection = {idx} if idx is not None else set()
        self._mark_dirty()

    def remove_selected(self) -> None:
        if not self._selection:
            return
        # Don't remove locked points
        removable = {i for i in self._selection
                     if not self.is_locked(self._actions[i].at)}
        if not removable:
            return
        self._snapshot(f"Delete {len(removable)} points")
        self._actions = [a for i, a in enumerate(self._actions) if i not in removable]
        self._selection.clear()
        self._mark_dirty()

    def move_action(self, idx: int, new_at: int, new_pos: int) -> int:
        """Move a single action to a new time/position. No snapshot — caller must snapshot on drag start.
        Returns the new index of the moved action after re-sort, or -1 if not moved."""
        if idx < 0 or idx >= len(self._actions):
            return -1
        if self.is_locked(self._actions[idx].at):
            return idx
        new_pos = max(0, min(100, new_pos))
        self._actions[idx] = FunscriptAction(new_at, new_pos)
        self._sort_actions()
        # Find the new index after sort
        new_idx = self._find_action_index(new_at, new_pos)
        self._dirty = True
        self._multi_axis_stale = True
        self._version += 1
        self.changed.emit()
        return new_idx if new_idx is not None else -1

    def _find_action_index(self, at: int, pos: int) -> int | None:
        """Find the index of the action at (at, pos) using binary search on time."""
        lo = bisect.bisect_left(self._actions, at, key=lambda a: a.at)
        for i in range(lo, len(self._actions)):
            a = self._actions[i]
            if a.at != at:
                break
            if a.pos == pos:
                return i
        return None

    def begin_drag(self) -> None:
        """Take a snapshot at the start of a drag operation."""
        self._snapshot("Move point")

    def move_selection_time(self, offset_ms: int) -> None:
        if not self._selection:
            return
        self._snapshot("Shift selection time")
        for i in self._selection:
            if i < len(self._actions) and not self.is_locked(self._actions[i].at):
                self._actions[i] = FunscriptAction(
                    max(0, self._actions[i].at + offset_ms),
                    self._actions[i].pos,
                )
        self._sort_actions()
        self._mark_dirty()

    def move_selection_position(self, offset: int) -> None:
        if not self._selection:
            return
        self._snapshot("Shift selection position")
        for i in self._selection:
            if i < len(self._actions) and not self.is_locked(self._actions[i].at):
                self._actions[i] = FunscriptAction(
                    self._actions[i].at,
                    max(0, min(100, self._actions[i].pos + offset)),
                )
        self._mark_dirty()

    # ── Transforms ──────────────────────────────────────────

    def invert_selection(self) -> None:
        if not self._selection:
            return
        self._snapshot("Invert selection")
        for i in self._selection:
            if i < len(self._actions) and not self.is_locked(self._actions[i].at):
                self._actions[i] = FunscriptAction(
                    self._actions[i].at,
                    100 - self._actions[i].pos,
                )
        self._mark_dirty()

    def invert_all(self) -> None:
        if not self._actions:
            return
        self._snapshot("Invert all")
        for i in range(len(self._actions)):
            if not self.is_locked(self._actions[i].at):
                self._actions[i] = FunscriptAction(
                    self._actions[i].at,
                    100 - self._actions[i].pos,
                )
        self._mark_dirty()

    def center_at(self, target_mean: float, selected_only: bool = False) -> None:
        """Shift points vertically so their mean equals *target_mean* (0-100).

        If *selected_only* is True and there is a selection, only those points
        are shifted; otherwise all unlocked points are shifted.
        """
        indices = sorted(self._selection) if (selected_only and self._selection) else list(range(len(self._actions)))
        unlocked = [i for i in indices if not self.is_locked(self._actions[i].at)]
        if not unlocked:
            return
        current_mean = sum(self._actions[i].pos for i in unlocked) / len(unlocked)
        offset = round(target_mean - current_mean)
        if offset == 0:
            return
        label = "selected" if (selected_only and self._selection) else "all"
        self._snapshot(f"Center {label} at {target_mean:.0f}")
        for i in unlocked:
            new_pos = max(0, min(100, self._actions[i].pos + offset))
            self._actions[i] = FunscriptAction(self._actions[i].at, new_pos)
        self._mark_dirty()

    def equalize_selection(self) -> None:
        """Redistribute selected actions evenly in time."""
        indices = sorted(self._selection)
        if len(indices) < 3:
            return
        unlocked = [i for i in indices if not self.is_locked(self._actions[i].at)]
        if len(unlocked) < 3:
            return
        self._snapshot("Equalize selection")
        first_t = self._actions[unlocked[0]].at
        last_t = self._actions[unlocked[-1]].at
        n = len(unlocked)
        for j, i in enumerate(unlocked):
            t = int(first_t + (last_t - first_t) * j / (n - 1))
            self._actions[i] = FunscriptAction(t, self._actions[i].pos)
        self._sort_actions()
        self._mark_dirty()

    def scale_selection(self, factor: float) -> None:
        """Scale selected positions around their midpoint."""
        if not self._selection:
            return
        unlocked = [i for i in self._selection if not self.is_locked(self._actions[i].at)]
        if not unlocked:
            return
        self._snapshot("Scale selection")
        positions = [self._actions[i].pos for i in unlocked]
        midpoint = sum(positions) / len(positions)
        for i in unlocked:
            new_pos = int(midpoint + (self._actions[i].pos - midpoint) * factor)
            self._actions[i] = FunscriptAction(
                self._actions[i].at,
                max(0, min(100, new_pos)),
            )
        self._mark_dirty()

    # ── Clipboard ───────────────────────────────────────────

    def copy_selection(self) -> None:
        if not self._selection:
            return
        sel = sorted(self._selection)
        self._clipboard = [FunscriptAction(self._actions[i].at, self._actions[i].pos) for i in sel]

    def cut_selection(self) -> None:
        if not self._selection:
            return
        self._snapshot("Cut selection")
        sel = sorted(self._selection)
        self._clipboard = [FunscriptAction(self._actions[i].at, self._actions[i].pos) for i in sel]
        removable = {i for i in sel if not self.is_locked(self._actions[i].at)}
        self._actions = [a for i, a in enumerate(self._actions) if i not in removable]
        self._selection.clear()
        self._mark_dirty()

    def paste_at(self, time_ms: int) -> None:
        """Paste clipboard relative to time_ms (offset from first clipboard action)."""
        if not self._clipboard:
            return
        self._snapshot("Paste")
        base_t = self._clipboard[0].at
        new_selection = set()
        for ca in self._clipboard:
            offset_t = ca.at - base_t
            new_at = time_ms + offset_t
            if self.is_locked(new_at):
                continue
            new_action = FunscriptAction(new_at, ca.pos)
            bisect.insort(self._actions, new_action, key=lambda a: a.at)
            idx = self._find_action_index(new_at, ca.pos)
            if idx is not None:
                new_selection.add(idx)
        self._selection = new_selection
        self._mark_dirty()

    def paste_exact(self) -> None:
        """Paste clipboard at original timestamps."""
        if not self._clipboard:
            return
        self._snapshot("Paste exact")
        new_selection = set()
        for ca in self._clipboard:
            if self.is_locked(ca.at):
                continue
            new_action = FunscriptAction(ca.at, ca.pos)
            bisect.insort(self._actions, new_action, key=lambda a: a.at)
            idx = self._find_action_index(ca.at, ca.pos)
            if idx is not None:
                new_selection.add(idx)
        self._selection = new_selection
        self._mark_dirty()

    # ── Locking ─────────────────────────────────────────────

    def lock_selection_region(self) -> None:
        """Lock the time range spanning the current selection."""
        if not self._selection:
            return
        sel_actions = self.selected_actions
        if not sel_actions:
            return
        start = min(a.at for a in sel_actions)
        end = max(a.at for a in sel_actions)
        self.lock_region(start, end)

    def lock_region(self, start_ms: int, end_ms: int) -> None:
        self._snapshot("Lock region")
        new_region = LockedRegion(min(start_ms, end_ms), max(start_ms, end_ms))
        self._locked_regions.append(new_region)
        self._merge_locked_regions()
        self._mark_dirty()

    def unlock_region(self, start_ms: int, end_ms: int) -> None:
        """Remove any locked region that overlaps with [start_ms, end_ms]."""
        self._snapshot("Unlock region")
        self._locked_regions = [
            r for r in self._locked_regions
            if r.end_ms < start_ms or r.start_ms > end_ms
        ]
        self._mark_dirty()

    def unlock_at(self, time_ms: int) -> None:
        """Remove the locked region containing the given time, if any."""
        region = next((r for r in self._locked_regions
                       if r.start_ms <= time_ms <= r.end_ms), None)
        if region is None:
            return
        self._snapshot("Unlock region")
        self._locked_regions.remove(region)
        self._mark_dirty()

    def clear_all_locks(self) -> None:
        if not self._locked_regions:
            return
        self._snapshot("Clear all locks")
        self._locked_regions.clear()
        self._mark_dirty()

    def lock_all_except_selection(self) -> None:
        """Lock everything OUTSIDE the current selection's time range."""
        if not self._selection:
            return
        self._snapshot("Lock All Except Selection")
        sel_actions = [self._actions[i] for i in sorted(self._selection) if i < len(self._actions)]
        if not sel_actions:
            return
        sel_start = min(a.at for a in sel_actions)
        sel_end = max(a.at for a in sel_actions)
        self._locked_regions.clear()
        if sel_start > 0:
            self._locked_regions.append(LockedRegion(0, sel_start - 1))
        max_at = self._actions[-1].at if self._actions else sel_end
        if sel_end < max_at:
            self._locked_regions.append(LockedRegion(sel_end + 1, max_at))
        self._mark_dirty()

    def get_locked_actions(self) -> list[FunscriptAction]:
        """Return actions that fall inside any locked region."""
        return [a for a in self._actions if self.is_locked(a.at)]

    def get_unlocked_gaps(self, duration_ms: int | None = None) -> list[tuple[int, int]]:
        """Return (start, end) ranges that are NOT locked."""
        if not self._locked_regions:
            end = duration_ms if duration_ms else (self._actions[-1].at if self._actions else 0)
            return [(0, end)]
        sorted_regions = sorted(self._locked_regions, key=lambda r: r.start_ms)
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for r in sorted_regions:
            if r.start_ms > cursor:
                gaps.append((cursor, r.start_ms))
            cursor = max(cursor, r.end_ms)
        end = duration_ms if duration_ms else (self._actions[-1].at if self._actions else cursor)
        if cursor < end:
            gaps.append((cursor, end))
        return gaps

    def _merge_locked_regions(self) -> None:
        """Merge overlapping locked regions."""
        if len(self._locked_regions) <= 1:
            return
        self._locked_regions.sort(key=lambda r: r.start_ms)
        merged: list[LockedRegion] = [self._locked_regions[0]]
        for r in self._locked_regions[1:]:
            last = merged[-1]
            if r.start_ms <= last.end_ms:
                merged[-1] = LockedRegion(last.start_ms, max(last.end_ms, r.end_ms))
            else:
                merged.append(r)
        self._locked_regions = merged

    # ── Undo / Redo ─────────────────────────────────────────

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._make_snapshot("redo"))
        snapshot = self._undo_stack.pop()
        self._actions = self._clone_actions(snapshot.actions)
        self._locked_regions = self._clone_locked_regions(snapshot.locked_regions)
        self._selection = set(snapshot.selection)
        self._set_state_flags()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._make_snapshot("undo"))
        snapshot = self._redo_stack.pop()
        self._actions = self._clone_actions(snapshot.actions)
        self._locked_regions = self._clone_locked_regions(snapshot.locked_regions)
        self._selection = set(snapshot.selection)
        self._set_state_flags()
        return True

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_description(self) -> str:
        if self._undo_stack:
            return self._undo_stack[-1].description
        return ""

    @property
    def redo_description(self) -> str:
        if self._redo_stack:
            return self._redo_stack[-1].description
        return ""

    # ── Pattern fill (used by Phase 7) ──────────────────────

    def fill_pattern(self, t_start_ms: int, t_end_ms: int,
                     pattern_actions: list[FunscriptAction]) -> None:
        """Replace actions in [t_start, t_end] with pattern-generated ones."""
        self._snapshot("Pattern Fill")
        # Remove non-locked actions in range
        self._actions = [
            a for a in self._actions
            if a.at < t_start_ms or a.at > t_end_ms or self.is_locked(a.at)
        ]
        # Insert new actions, skipping locked regions
        for a in pattern_actions:
            if not self.is_locked(a.at):
                bisect.insort(self._actions, a, key=lambda x: x.at)
        self._selection.clear()
        self._mark_dirty()
