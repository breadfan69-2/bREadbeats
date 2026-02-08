"""
Generate splash screen image for bREadbeats using PicTex.
Uses bbicon.png as background with stylized loading text overlay.
"""
from pictex import Canvas, Text, Row, Column, Shadow
from PIL import Image as PILImage

def create_splash_screen():
    """Create a splash screen with the icon and loading text."""
    
    # Load and resize the icon for splash screen (200px = 50% of original 400px)
    icon = PILImage.open("bbicon.png")
    icon_resized = icon.resize((200, int(200 * icon.height / icon.width)), PILImage.Resampling.LANCZOS)
    icon_resized.save("bbicon_splash.png")
    
    # Create the splash screen using PicTex
    # Dark gradient background with the icon and loading text
    canvas = (
        Canvas()
        .font_size(16)  # Smaller font for smaller splash
        .padding(20, 30)  # Reduced padding
        .background_color("#1a1a2e")  # Dark blue-gray background
        .color("#ffffff")  # White text
        .text_shadows(Shadow(offset=(1, 1), blur_radius=2, color="#000000"))
    )
    
    # Render the loading text
    text_image = canvas.render("Please wait, loading BeatTracker modules....")
    text_image.save("loading_text.png")
    
    # Now combine the icon and text into a final splash image using PIL
    text_pil = PILImage.open("loading_text.png")
    icon_pil = PILImage.open("bbicon_splash.png")
    
    # Calculate splash dimensions (50% smaller)
    splash_width = max(icon_pil.width, text_pil.width) + 40  # Reduced padding
    splash_height = icon_pil.height + text_pil.height + 50  # Icon + text + spacing
    
    # Create splash background
    splash = PILImage.new("RGBA", (splash_width, splash_height), (26, 26, 46, 255))  # #1a1a2e
    
    # Center and paste icon
    icon_x = (splash_width - icon_pil.width) // 2
    icon_y = 15  # Reduced top margin
    
    # Handle transparency for icon
    if icon_pil.mode == 'RGBA':
        splash.paste(icon_pil, (icon_x, icon_y), icon_pil)
    else:
        splash.paste(icon_pil, (icon_x, icon_y))
    
    # Center and paste text below icon
    text_x = (splash_width - text_pil.width) // 2
    text_y = icon_y + icon_pil.height + 15  # Reduced spacing
    
    if text_pil.mode == 'RGBA':
        splash.paste(text_pil, (text_x, text_y), text_pil)
    else:
        splash.paste(text_pil, (text_x, text_y))
    
    # Save final splash
    splash.save("splash_screen.png")
    print(f"Splash screen created: {splash_width}x{splash_height}")
    print("Saved to: splash_screen.png")

if __name__ == "__main__":
    create_splash_screen()
