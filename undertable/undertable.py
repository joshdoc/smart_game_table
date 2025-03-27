import arcade

# Constants for resizing graphics and player speed
TITLE_SCALE = 2.0
PLAYER_SCALE = 4.0
BOUNDARY_SCALE = 20.0
BOSS_SCALE = 1.2
PLAYER_SPEED = 10

class UndertableGame(arcade.Window):
    def __init__(self):
        # Set fullscreen to True here
        super().__init__(fullscreen=True)
        self.width, self.height = self.get_size()
        arcade.set_background_color(arcade.color.BLACK)
        
        # Load assets
        self.title_img = arcade.load_texture("graphics/undertable_title.png")
        self.player_img = arcade.load_texture("graphics/player.png")
        self.boundary_img = arcade.load_texture("graphics/boundary.png")
        self.boss_img = arcade.load_texture("graphics/boss.png")
        
        # Resize graphics
        self.title_width = int(self.title_img.width * TITLE_SCALE)
        self.title_height = int(self.title_img.height * TITLE_SCALE)
        self.player_width = int(self.player_img.width * PLAYER_SCALE)
        self.player_height = int(self.player_img.height * PLAYER_SCALE)
        self.boundary_width = int(self.boundary_img.width * BOUNDARY_SCALE)
        self.boundary_height = int(self.boundary_img.height * BOUNDARY_SCALE)
        self.boss_width = int(self.boss_img.width * BOSS_SCALE)
        self.boss_height = int(self.boss_img.height * BOSS_SCALE)
        
        # Position graphics
        self.title_x = self.width // 2
        self.title_y = self.height // 2
        self.player_x = self.width // 2
        self.player_y = self.height // 3 * 2
        self.boundary_x = self.width // 2
        self.boundary_y = self.height // 3 * 2
        self.boss_x = self.width // 2
        self.boss_y = self.height // 4
        
        self.game_state = "menu"
    
    def on_draw(self):
        arcade.start_render()
        if self.game_state == "menu":
            arcade.draw_texture_rectangle(self.title_x, self.title_y, self.title_width, self.title_height, self.title_img)
        elif self.game_state == "play":
            arcade.draw_texture_rectangle(self.boundary_x, self.boundary_y, self.boundary_width, self.boundary_height, self.boundary_img)
            arcade.draw_texture_rectangle(self.boss_x, self.boss_y, self.boss_width, self.boss_height, self.boss_img)
            arcade.draw_texture_rectangle(self.player_x, self.player_y, self.player_width, self.player_height, self.player_img)
    
    def on_mouse_press(self, x, y, button, modifiers):
        print(f"Mouse pressed at: {x}, {y}")
        title_left = self.title_x - self.title_width // 2
        title_right = self.title_x + self.title_width // 2
        title_bottom = self.title_y - self.title_height // 2
        title_top = self.title_y + self.title_height // 2
        
        if title_left <= x <= title_right and title_bottom <= y <= title_top and self.game_state == "menu":
            print("Title clicked! Switching to play mode.")
            self.game_state = "play"
        else:
            print("Click outside title bounds.")
    
    def on_key_press(self, key, modifiers):
        if key == arcade.key.ESCAPE:
            self.close()
    
    def on_update(self, delta_time):
        keys = arcade.get_keys()
        if self.game_state == "play":
            if arcade.key.LEFT in keys and self.player_x > self.boundary_x - self.boundary_width // 2 + 10:
                self.player_x -= PLAYER_SPEED
            if arcade.key.RIGHT in keys and self.player_x < self.boundary_x + self.boundary_width // 2 - 10:
                self.player_x += PLAYER_SPEED
            if arcade.key.UP in keys and self.player_y < self.boundary_y + self.boundary_height // 2 - 10:
                self.player_y += PLAYER_SPEED
            if arcade.key.DOWN in keys and self.player_y > self.boundary_y - self.boundary_height // 2 + 10:
                self.player_y -= PLAYER_SPEED

if __name__ == "__main__":
    game = UndertableGame()
    arcade.run()
