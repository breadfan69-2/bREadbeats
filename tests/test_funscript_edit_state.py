from __future__ import annotations

import unittest

from PyQt6.QtWidgets import QApplication

from funscript_edit_state import FunscriptEditState, LockedRegion
from pmv_funscript_io import FunscriptAction


class TestFunscriptEditState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QApplication.instance() or QApplication([])

    def test_undo_redo_restore_clean_dirty_state(self):
        state = FunscriptEditState()
        state.load_actions([
            FunscriptAction(0, 10),
            FunscriptAction(500, 50),
        ])

        self.assertFalse(state.dirty)

        state.select_index(1)
        state.move_selection_position(10)
        self.assertTrue(state.dirty)

        self.assertTrue(state.undo())
        self.assertFalse(state.dirty)
        self.assertEqual([(action.at, action.pos) for action in state.actions], [(0, 10), (500, 50)])

        self.assertTrue(state.redo())
        self.assertTrue(state.dirty)
        self.assertEqual([(action.at, action.pos) for action in state.actions], [(0, 10), (500, 60)])

    def test_load_actions_clears_existing_locks(self):
        state = FunscriptEditState()
        state.load_actions([FunscriptAction(100, 25), FunscriptAction(200, 75)])
        state.lock_region(100, 200)
        self.assertEqual(state.locked_regions, [LockedRegion(100, 200)])

        state.load_actions([FunscriptAction(300, 40)])
        self.assertEqual(state.locked_regions, [])
        self.assertFalse(state.dirty)

    def test_lock_all_except_selection_keeps_boundary_points_unlocked(self):
        state = FunscriptEditState()
        state.load_actions([
            FunscriptAction(100, 10),
            FunscriptAction(200, 20),
            FunscriptAction(300, 30),
        ])

        state.select_range(200, 300)
        state.lock_all_except_selection()

        self.assertTrue(state.is_locked(100))
        self.assertFalse(state.is_locked(200))
        self.assertFalse(state.is_locked(300))


if __name__ == "__main__":
    unittest.main()