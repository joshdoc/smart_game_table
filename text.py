import pyglet

# Add the custom font file.
pyglet.font.add_file("graphics/PressStart2P.ttf")

# Create a window.
window = pyglet.window.Window(800, 600, caption="Text Display Test")

# Create a label using the custom font.
# Adjust the font_name if needed if your system doesn't pick it up.
label = pyglet.text.Label(
    "Score: 0",
    font_name="Press Start 2P",
    font_size=32,
    x=window.width // 2,
    y=window.height // 2,
    anchor_x="center",
    anchor_y="center",
    color=(255, 255, 255, 255)
)

@window.event
def on_draw():
    window.clear()
    label.draw()

pyglet.app.run()
