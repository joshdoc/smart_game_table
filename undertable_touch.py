####################################################################################################
# IMPORTS                                                                                          #
####################################################################################################

import pyglet
import random
import math
import time
from enum import Enum
import numpy as np
from pyglet.window import key, mouse
import cv as cv
from dataclasses import dataclass
from sgt_types import Centroid, DetectedCentroids, Loop_Result_t

####################################################################################################
# CONSTANTS                                                                                        #
####################################################################################################

BASE_PROBE_PROJECTILE_SPEED_RANGE    = (250, 350)
BASE_PROBE_PROJECTILE_DISTANCE_RANGE = (400, 500)
BASE_PROBE_DELAY_RANGE               = (0.3, 0.7)
BASE_PROBE_VARIABILITY_RANGE         = (10, 30)
BASE_PROBE_SPAWN_RATE                = 1.0 
PROBE_FADE_DURATION                  = 0.3
PROBE_BOUNDARY_INSET                 = 50

PLAYER_SPEED                         = 500

PLAYER_COLLISION_SCALE               = 0.8
PROBE_COLLISION_SCALE                = 0.8

GRAVITY                              = 2000
JUMP_INITIAL_VELOCITY                = 1000
DRIFT_SPEED                          = 400
DODGE_DURATION                       = 10.0

BASE_PULSE_SPAWN_RATE                = 0.4
BASE_PULSE_SPEED                     = 300
PULSE_THIN_PROBABILITY               = 0.5

SHIELD_IDLE_DURATION                 = 10.0
SHIELD_DRIFT_SPEED                   = 400
SHIELD_RADIUS                        = 80
SHIELD_ROTATION_SPEED                = 700
SHIELD_EXIT_DURATION                 = 10.0

BASE_CAPACITOR_SPAWN_RATE            = 0.75
BASE_CAPACITOR_SPEED                 = 200
CAPACITOR_ROTATION_SPEED             = 180

####################################################################################################
# GLOBALS                                                                                          #
####################################################################################################

# Enable mouse input

mouse_debug = False

# Score-dependent globals

PROBE_PROJECTILE_SPEED_RANGE    = BASE_PROBE_PROJECTILE_SPEED_RANGE
PROBE_PROJECTILE_DISTANCE_RANGE = BASE_PROBE_PROJECTILE_DISTANCE_RANGE
PROBE_DELAY_RANGE               = BASE_PROBE_DELAY_RANGE
PROBE_VARIABILITY_RANGE         = BASE_PROBE_VARIABILITY_RANGE
PROBE_SPAWN_RATE                = BASE_PROBE_SPAWN_RATE

PULSE_SPAWN_RATE                = BASE_PULSE_SPAWN_RATE
PULSE_SPEED                     = BASE_PULSE_SPEED

CAPACITOR_SPAWN_RATE            = BASE_CAPACITOR_SPAWN_RATE
CAPACITOR_SPEED                 = BASE_CAPACITOR_SPEED

# Game state globals

game_state        = "menu"  # "menu", "play", "drift", "jump", "shield", "end"
fading            = False
alpha             = 255
player_alpha      = 0
boundary_alpha    = 0
boss_alpha        = 0
player_lives      = 3
invincible        = False
invincibility_duration = 1.0
flash_interval    = 0.1

probes            = []
pulse_waves       = []
capacitors        = []
dodge_timer       = 0.0
player_vy         = 0.0
jump_charging     = False
jump_mode_timer   = 0.0
shield_current_angle = 0.0
shield_target_angle  = 0.0
shield_mode_timer    = 0.0

# Sprites initialized later in init()
title_image             = None
player_image            = None
player_jump_image       = None
player_shield_image     = None
boundary_image          = None
boss_left_image         = None
boss_center_image       = None
boss_right_image        = None
boss_body_image         = None
health_3_image          = None
health_2_image          = None
health_1_image          = None
health_3_jump_image     = None
health_2_jump_image     = None
health_1_jump_image     = None
health_3_shield_image   = None
health_2_shield_image   = None
health_1_shield_image   = None
red_probe_image         = None
black_probe_image       = None
oscilloscope_image      = None
large_oscilloscope_image= None
shield_image            = None
pulse_thin_image        = None
pulse_wide_image        = None
capacitor_image         = None

global title_sprite, player_sprite, boundary_sprite, boss_sprite
global boss_body_sprite, large_oscilloscope_sprite
global health_sprite_normal, health_sprite_jump, health_sprite_shield
global shield_sprite

original_score_x          = None
original_score_y          = None
window                   = None
exit_game               = False
game_over               = False
score                   = 0
score_label             = None
title_sprite             = None

####################################################################################################
# CLASSES                                                                                          #
####################################################################################################

class Probe:
    def __init__(self, x, y, direction, speed, delay):
        self.image = random.choice([red_probe_image, black_probe_image])
        self.sprite = pyglet.sprite.Sprite(self.image, x=x, y=y)
        self.sprite.rotation = -math.degrees(direction)
        self.sprite.opacity = 0
        self.state = "fading_in"
        self.timer = 0.0
        self.vx = speed * math.cos(direction)
        self.vy = speed * math.sin(direction)
        self.delay = delay

    def update(self, dt):
        if self.state == "fading_in":
            inc = 255 * dt / PROBE_FADE_DURATION
            self.sprite.opacity = int(min(255, self.sprite.opacity + inc))
            if self.sprite.opacity >= 255:
                self.state = "waiting"
                self.timer = self.delay

        elif self.state == "waiting":
            self.timer -= dt
            if self.timer <= 0:
                self.state = "moving"

        elif self.state == "moving":
            self.sprite.x += self.vx * dt
            self.sprite.y += self.vy * dt
            lb = boundary_sprite.x - boundary_sprite.width/2 + PROBE_BOUNDARY_INSET
            rb = boundary_sprite.x + boundary_sprite.width/2 - PROBE_BOUNDARY_INSET
            bb = boundary_sprite.y - boundary_sprite.height/2 + PROBE_BOUNDARY_INSET
            tb = boundary_sprite.y + boundary_sprite.height/2 - PROBE_BOUNDARY_INSET
            if (self.sprite.x < lb or self.sprite.x > rb or
                self.sprite.y < bb or self.sprite.y > tb):
                self.state = "fading_out"

        elif self.state == "fading_out":
            dec = 255 * dt / PROBE_FADE_DURATION
            self.sprite.opacity = int(max(0, self.sprite.opacity - dec))
            if self.sprite.opacity <= 0:
                return True
        return False

    def draw(self):
        self.sprite.draw()

    def collides_with_player(self):
        pw = player_sprite.width * PLAYER_COLLISION_SCALE
        ph = player_sprite.height * PLAYER_COLLISION_SCALE
        pl = player_sprite.x - pw/2
        pr = player_sprite.x + pw/2
        pb = player_sprite.y - ph/2
        pt = player_sprite.y + ph/2

        iw = self.sprite.width * PROBE_COLLISION_SCALE
        ih = self.sprite.height * PROBE_COLLISION_SCALE
        il = self.sprite.x - iw/2
        ir = self.sprite.x + iw/2
        ib = self.sprite.y - ih/2
        it = self.sprite.y + ih/2

        return (pl < ir and pr > il and pb < it and pt > ib)

class PulseWave:
    def __init__(self, img, x, y, speed):
        self.sprite = pyglet.sprite.Sprite(img, x=x, y=y)
        self.sprite.opacity = 255
        self.speed = speed

    def update(self, dt):
        self.sprite.x += self.speed * dt
        rb = boundary_sprite.x + boundary_sprite.width/2 - PROBE_BOUNDARY_INSET
        return self.sprite.x > rb

    def draw(self):
        self.sprite.draw()

    def collides_with_player(self):
        pw = player_sprite.width * PLAYER_COLLISION_SCALE
        ph = player_sprite.height * PLAYER_COLLISION_SCALE
        pl = player_sprite.x - pw/2
        pr = player_sprite.x + pw/2
        pb = player_sprite.y - ph/2
        pt = player_sprite.y + ph/2

        iw = self.sprite.width
        ih = self.sprite.height
        il = self.sprite.x - iw/2
        ir = self.sprite.x + iw/2
        ib = self.sprite.y - ih/2
        it = self.sprite.y + ih/2

        return (pl < ir and pr > il and pb < it and pt > ib)

class Capacitor:
    def __init__(self, direction):
        if direction == "up":
            sx = boundary_sprite.x
            sy = boundary_sprite.y + boundary_sprite.height/2 - PROBE_BOUNDARY_INSET
        elif direction == "down":
            sx = boundary_sprite.x
            sy = boundary_sprite.y - boundary_sprite.height/2 + PROBE_BOUNDARY_INSET
        elif direction == "left":
            sx = boundary_sprite.x - boundary_sprite.width/2 + PROBE_BOUNDARY_INSET
            sy = boundary_sprite.y
        else:
            sx = boundary_sprite.x + boundary_sprite.width/2 - PROBE_BOUNDARY_INSET
            sy = boundary_sprite.y

        self.sprite = pyglet.sprite.Sprite(capacitor_image, x=sx, y=sy)
        self.sprite.opacity = 255

        dx = player_sprite.x - sx
        dy = player_sprite.y - sy
        mag = math.hypot(dx, dy) or 1
        self.vx = (dx/mag) * CAPACITOR_SPEED
        self.vy = (dy/mag) * CAPACITOR_SPEED
        self.rotation_speed = CAPACITOR_ROTATION_SPEED

    def update(self, dt):
        self.sprite.x += self.vx * dt
        self.sprite.y += self.vy * dt
        self.sprite.rotation = (self.sprite.rotation + self.rotation_speed * dt) % 360

        lb = boundary_sprite.x - boundary_sprite.width/2 + PROBE_BOUNDARY_INSET
        rb = boundary_sprite.x + boundary_sprite.width/2 - PROBE_BOUNDARY_INSET
        bb = boundary_sprite.y - boundary_sprite.height/2 + PROBE_BOUNDARY_INSET
        tb = boundary_sprite.y + boundary_sprite.height/2 - PROBE_BOUNDARY_INSET

        return (self.sprite.x < lb or self.sprite.x > rb or
                self.sprite.y < bb or self.sprite.y > tb)

    def draw(self):
        self.sprite.draw()

    def collides_with_shield(self):
        cl = self.sprite.x - self.sprite.width/2
        cr = self.sprite.x + self.sprite.width/2
        cb = self.sprite.y - self.sprite.height/2
        ct = self.sprite.y + self.sprite.height/2

        sl = shield_sprite.x - shield_sprite.width/2
        sr = shield_sprite.x + shield_sprite.width/2
        sb = shield_sprite.y - shield_sprite.height/2
        st = shield_sprite.y + shield_sprite.height/2

        return not (cr < sl or cl > sr or ct < sb or cb > st)

    def collides_with_player(self):
        pw = player_sprite.width * PLAYER_COLLISION_SCALE
        ph = player_sprite.height * PLAYER_COLLISION_SCALE
        pl = player_sprite.x - pw/2
        pr = player_sprite.x + pw/2
        pb = player_sprite.y - ph/2
        pt = player_sprite.y + ph/2

        cw = self.sprite.width
        ch = self.sprite.height
        cl = self.sprite.x - cw/2
        cr = self.sprite.x + cw/2
        cb = self.sprite.y - ch/2
        ct = self.sprite.y + ch/2

        return (pl < cr and pr > cl and pb < ct and pt > cb)

####################################################################################################
# FUNCTIONS                                                                                        #
####################################################################################################

def spawn_probe(dt):
    global probes
    if game_state != "play":
        return
    speed      = random.uniform(*PROBE_PROJECTILE_SPEED_RANGE)
    distance   = random.uniform(*PROBE_PROJECTILE_DISTANCE_RANGE)
    delay      = random.uniform(*PROBE_DELAY_RANGE)
    variability= random.uniform(*PROBE_VARIABILITY_RANGE)

    angle = math.radians(random.uniform(0,360))
    sx = player_sprite.x + distance*math.cos(angle)
    sy = player_sprite.y + distance*math.sin(angle)

    lb = boundary_sprite.x - boundary_sprite.width/2 + PROBE_BOUNDARY_INSET
    rb = boundary_sprite.x + boundary_sprite.width/2 - PROBE_BOUNDARY_INSET
    bb = boundary_sprite.y - boundary_sprite.height/2 + PROBE_BOUNDARY_INSET
    tb = boundary_sprite.y + boundary_sprite.height/2 - PROBE_BOUNDARY_INSET

    sx = max(min(sx, rb), lb)
    sy = max(min(sy, tb), bb)

    ideal = math.atan2(player_sprite.y - sy, player_sprite.x - sx)
    final = ideal + math.radians(random.uniform(-variability, variability))

    probes.append(Probe(sx, sy, final, speed, delay))

def spawn_pulse(dt):
    if game_state != "jump":
        return
    sx = boundary_sprite.x - boundary_sprite.width/2 + PROBE_BOUNDARY_INSET
    sy = boundary_sprite.y - boundary_sprite.height/2 + PROBE_BOUNDARY_INSET
    img= pulse_thin_image if random.random() < PULSE_THIN_PROBABILITY else pulse_wide_image
    pulse_waves.append(PulseWave(img, sx, sy, PULSE_SPEED))

def spawn_capacitor(dt):
    if game_state != "shield":
        return
    capacitors.append(Capacitor(random.choice(["up","down","left","right"])))

# ----------------------------
# Dynamic Scaling Hook
# ----------------------------
def update_config():
    global PROBE_PROJECTILE_SPEED_RANGE, PROBE_SPAWN_RATE
    global PULSE_SPAWN_RATE, PULSE_SPEED
    global CAPACITOR_SPAWN_RATE, CAPACITOR_SPEED

    lo, hi = BASE_PROBE_PROJECTILE_SPEED_RANGE
    PROBE_PROJECTILE_SPEED_RANGE = (lo + score*15, hi + score*15)
    PROBE_SPAWN_RATE = BASE_PROBE_SPAWN_RATE + 0.1*score

    PULSE_SPAWN_RATE = BASE_PULSE_SPAWN_RATE
    PULSE_SPEED      = BASE_PULSE_SPEED + 25*score

    CAPACITOR_SPAWN_RATE = BASE_CAPACITOR_SPAWN_RATE + 0.1*score
    CAPACITOR_SPEED      = BASE_CAPACITOR_SPEED + 25*score

    pyglet.clock.unschedule(spawn_probe)
    pyglet.clock.schedule_interval(spawn_probe,    1/PROBE_SPAWN_RATE)
    pyglet.clock.unschedule(spawn_pulse)
    pyglet.clock.schedule_interval(spawn_pulse,    1/PULSE_SPAWN_RATE)
    pyglet.clock.unschedule(spawn_capacitor)
    pyglet.clock.schedule_interval(spawn_capacitor,1/CAPACITOR_SPAWN_RATE)

    # ----------------------------
# Health Sprite Update
# ----------------------------
def update_health_sprite():
    if player_lives == 3:
        health_sprite_normal.image = health_3_image
        health_sprite_jump.image   = health_3_jump_image
        health_sprite_shield.image = health_3_shield_image
    elif player_lives == 2:
        health_sprite_normal.image = health_2_image
        health_sprite_jump.image   = health_2_jump_image
        health_sprite_shield.image = health_2_shield_image
    else:
        health_sprite_normal.image = health_1_image
        health_sprite_jump.image   = health_1_jump_image
        health_sprite_shield.image = health_1_shield_image

# ----------------------------
# Main Update Loop
# ----------------------------
def update(dt):
    global alpha, fading, game_state, player_alpha, boundary_alpha, boss_alpha
    global dodge_timer, player_vy, player_target_x, player_target_y
    global jump_charging, jump_mode_timer, shield_current_angle
    global shield_target_angle, shield_mode_timer, score

    # MENU → PLAY fade
    if game_state == "menu":
        if fading:
            alpha = max(0, alpha - 10)
            if alpha == 0:
                game_state = "play"
                fading = False

    # PLAY logic
    elif game_state == "play":
        if not invincible:
            player_alpha = min(255, player_alpha + 10)
        boundary_alpha = min(255, boundary_alpha + 10)
        boss_alpha     = min(255, boss_alpha + 10)

        # move player
        dx = player_target_x - player_sprite.x
        dy = player_target_y - player_sprite.y
        dist = math.hypot(dx, dy)
        if dist:
            step = PLAYER_SPEED * dt
            if step >= dist:
                player_sprite.x, player_sprite.y = player_target_x, player_target_y
            else:
                player_sprite.x += dx/dist * step
                player_sprite.y += dy/dist * step

        # boss pose
        lb = boundary_sprite.x - boundary_sprite.width/2 + 25
        rb = boundary_sprite.x + boundary_sprite.width/2 - 25
        region = (rb - lb)/3
        if player_sprite.x < lb + region:
            boss_sprite.image = boss_left_image
        elif player_sprite.x < lb + 2*region:
            boss_sprite.image = boss_center_image
        else:
            boss_sprite.image = boss_right_image

        # probes
        removal = []
        for p in probes:
            if p.update(dt) or p.collides_with_player():
                if p.collides_with_player():
                    player_hit()
                removal.append(p)
        for p in removal:
            probes.remove(p)

        # dodge timer
        dodge_timer += dt
        if dodge_timer >= DODGE_DURATION:
            pyglet.clock.unschedule(spawn_probe)
            probes.clear()
            game_state = "drift"

    # DRIFT → JUMP
    elif game_state == "drift":
        ground = boundary_sprite.y - boundary_sprite.height/2 + 25 + player_sprite.height/2
        dx = boundary_sprite.x - player_sprite.x
        dy = ground - player_sprite.y
        dist = math.hypot(dx, dy)
        step = DRIFT_SPEED * dt
        if dist and step < dist:
            player_sprite.x += dx/dist * step
            player_sprite.y += dy/dist * step
        else:
            player_sprite.x = boundary_sprite.x
            player_sprite.y = ground
            player_sprite.image = player_jump_image
            update_health_sprite()
            score += 1
            score_label.text = str(score)
            update_config()
            game_state = "jump"
            player_vy = 0.0
            jump_mode_timer = 0.0
            pulse_waves.clear()

    # JUMP logic
    elif game_state == "jump":
        player_sprite.x = boundary_sprite.x
        ground = boundary_sprite.y - boundary_sprite.height/2 + 25 + player_sprite.height/2
        ceiling = boundary_sprite.y + 200
        boss_sprite.image = boss_center_image

        eff_gravity = GRAVITY * 0.5 if jump_charging else GRAVITY
        player_vy   -= eff_gravity * dt
        player_sprite.y += player_vy * dt

        # clamp
        if player_sprite.y > ceiling:
            player_sprite.y = ceiling
            player_vy = 0
            jump_charging = False
        if player_sprite.y < ground:
            player_sprite.y = ground
            player_vy = 0
            jump_charging = False

        # pulses
        removal = []
        for pw in pulse_waves:
            if pw.update(dt) or pw.collides_with_player():
                if pw.collides_with_player():
                    player_hit()
                removal.append(pw)
        for pw in removal:
            pulse_waves.remove(pw)

        jump_mode_timer += dt
        if jump_mode_timer >= SHIELD_IDLE_DURATION:
            game_state = "shield"
            player_sprite.image = player_shield_image
            update_health_sprite()
            shield_current_angle = 0.0
            shield_target_angle  = 0.0
            shield_mode_timer    = 0.0
            pulse_waves.clear()
            score += 1
            score_label.text = str(score)
            update_config()

    # SHIELD logic
    elif game_state == "shield":
        dx = boundary_sprite.x - player_sprite.x
        dy = boundary_sprite.y - player_sprite.y
        dist = math.hypot(dx, dy)
        step = SHIELD_DRIFT_SPEED * dt
        if dist and step < dist:
            player_sprite.x += dx/dist * step
            player_sprite.y += dy/dist * step
        else:
            player_sprite.x = boundary_sprite.x
            player_sprite.y = boundary_sprite.y

        boss_sprite.image = boss_center_image

        diff = ((shield_target_angle - shield_current_angle + 180) % 360) - 180
        rot_step = SHIELD_ROTATION_SPEED * dt
        if abs(diff) <= rot_step:
            shield_current_angle = shield_target_angle
        else:
            shield_current_angle = (shield_current_angle + rot_step * (1 if diff>0 else -1)) % 360

        rad = math.radians(shield_current_angle)
        shield_sprite.x = player_sprite.x + SHIELD_RADIUS * math.cos(rad)
        shield_sprite.y = player_sprite.y + SHIELD_RADIUS * math.sin(rad)
        rot_face = math.degrees(math.atan2(
            shield_sprite.y - player_sprite.y,
            shield_sprite.x - player_sprite.x
        ))
        shield_sprite.rotation = -rot_face % 360

        # capacitors
        removal = []
        for c in capacitors:
            if c.update(dt) or c.collides_with_player() or c.collides_with_shield():
                if c.collides_with_player():
                    player_hit()
                removal.append(c)
        for c in removal:
            capacitors.remove(c)

        shield_mode_timer += dt
        if shield_mode_timer >= SHIELD_EXIT_DURATION:
            probes.clear()
            pulse_waves.clear()
            capacitors.clear()
            game_state = "play"
            player_sprite.image = player_image
            update_health_sprite()
            dodge_timer = shield_mode_timer = 0.0
            score += 1
            score_label.text = str(score)
            update_config()

# ----------------------------
# Flash & Invincibility
# ----------------------------
def flash_player(dt):
    player_sprite.opacity = 50 if player_sprite.opacity == 255 else 255

def end_invincibility(dt):
    global invincible
    pyglet.clock.unschedule(flash_player)
    player_sprite.opacity = 255
    invincible = False

def player_hit():
    global player_lives, invincible
    if invincible or player_lives <= 0:
        return
    player_lives -= 1
    update_health_sprite()
    invincible = True
    pyglet.clock.schedule_interval(flash_player, flash_interval)
    pyglet.clock.schedule_once(end_invincibility, invincibility_duration)
    if player_lives == 0:
        end_game()

# ----------------------------
# End‐game and Reset Helpers
# ----------------------------
def end_game():
    global game_state
    game_state = "end"
    pyglet.clock.unschedule(spawn_probe)
    pyglet.clock.unschedule(spawn_pulse)
    pyglet.clock.unschedule(spawn_capacitor)
    score_label.x = WIDTH//2 - 180
    score_label.y = HEIGHT//2 + 50
    score_label.font_size = 200

def reset_game():
    global game_state, player_lives, invincible, score
    global alpha, player_alpha, boundary_alpha, boss_alpha
    global dodge_timer, jump_mode_timer, shield_mode_timer
    global player_lives, invincible, score
    global temp_score

    # restore HUD
    score_label.x = original_score_x
    score_label.y = original_score_y
    score_label.font_size = 50

    player_sprite.image = player_image
    player_sprite.opacity = 255

    boss_body_sprite.opacity = 0

    game_state = "menu"
    alpha = 255
    player_alpha = boundary_alpha = boss_alpha = 0
    player_lives = 3
    update_health_sprite()
    temp_score = score
    score = 0
    score_label.text = str(score)
    invincible = False
    probes.clear()
    pulse_waves.clear()
    capacitors.clear()
    dodge_timer = jump_mode_timer = shield_mode_timer = 0.0
    update_config()

####################################################################################################
# EVENT HANDLERS                                                                                   #
####################################################################################################

def on_mouse_press(x, y, button, modifiers):
    global fading, player_target_x, player_target_y, jump_charging, player_vy, shield_target_angle
    global exit_game
    if game_state == "end":
        exit_game = True
        return
    if game_state == "menu":
        if (abs(x - title_sprite.x) < title_sprite.width/2 and
            abs(y - title_sprite.y) < title_sprite.height/2):
            fading = True
    elif game_state == "play":
        lb = boundary_sprite.x - boundary_sprite.width/2 + 25 + player_sprite.width/2
        rb = boundary_sprite.x + boundary_sprite.width/2 - 25 - player_sprite.width/2
        bb = boundary_sprite.y - boundary_sprite.height/2 + 25 + player_sprite.height/2
        tb = boundary_sprite.y + boundary_sprite.height/2 - 25 - player_sprite.height/2
        player_target_x = max(min(x, rb), lb)
        player_target_y = max(min(HEIGHT - y, tb), bb)
    elif game_state == "jump":
        ground = boundary_sprite.y - boundary_sprite.height/2 + 25 + player_sprite.height/2
        if abs(player_sprite.y - ground) < 1e-3:
            jump_charging = True
            player_vy = JUMP_INITIAL_VELOCITY
    elif game_state == "shield":
        dx, dy = x - player_sprite.x, HEIGHT - y - player_sprite.y
        if abs(dx) >= abs(dy):
            shield_target_angle = 0 if dx>0 else 180
        else:
            shield_target_angle = 90 if dy>0 else 270

def on_mouse_release(x, y, button, modifiers):
    global jump_charging
    if game_state == "jump":
        jump_charging = False

def on_draw():
    window.clear()
    if game_state == "end":
        large_oscilloscope_sprite.draw()
        score_label.draw()
        return

    if game_state == "menu":
        title_sprite.opacity = alpha
        title_sprite.draw()
        return

    boundary_sprite.opacity = boundary_alpha
    boss_sprite.opacity     = boss_alpha
    boss_body_sprite.opacity    = boss_alpha
    boundary_sprite.draw()
    boss_body_sprite.draw()
    boss_sprite.draw()
    player_sprite.draw()

    if game_state == "shield":
        health_sprite_shield.draw()
        for c in capacitors:
            c.draw()
        shield_sprite.draw()
    elif game_state == "jump":
        health_sprite_jump.draw()
    else:
        health_sprite_normal.draw()

    for p in probes:
        p.draw()
    if game_state == "jump":
        for pw in pulse_waves:
            pw.draw()

    oscilloscope_sprite.draw()
    score_label.draw()

####################################################################################################
# INITIALIZATION                                                                                   #
####################################################################################################

def init(_=None):
    global exit_game, game_over, window
    global score_label, oscilloscope_sprite, title_sprite, player_sprite, boundary_sprite, boss_sprite, boss_body_sprite, large_oscilloscope_sprite
    global health_sprite_jump, health_sprite_normal, health_sprite_shield, boss_left_image, boss_right_image
    global player_target_x, player_target_y, boss_center_image, red_probe_image, black_probe_image, shield_sprite
    global player_jump_image, player_shield_image, health_2_image, health_1_image, health_2_jump_image, health_2_shield_image, health_1_shield_image
    global WIDTH, HEIGHT
    global title_image, player_image, player_shield_image, boundary_image, boss_left_image, boss_center_image, boss_right_image, boss_body_image, health_3_image, health_2_image, health_1_image, health_3_jump_image, health_2_jump_image, health_1_jump_image, health_3_shield_image, health_2_shield_image, health_1_shield_image, red_probe_image, black_probe_image, oscilloscope_image, large_oscilloscope_image, shield_image, pulse_thin_image, pulse_wide_image, capacitor_image         

    exit_game = False
    game_over = False

    window = pyglet.window.Window(fullscreen=True)
    WIDTH, HEIGHT = window.width, window.height
    window.set_caption("Undertable")

    # Load font
    pyglet.font.add_file("graphics/PressStart2P.ttf")

    # Score & HUD

    score = 0
    score_label = pyglet.text.Label(
        text="0",
        font_name="Press Start 2P",
        font_size=50,
        x=965, y=HEIGHT - 230,
        anchor_x="center", anchor_y="center",
        color=(255, 255, 255, 255)
    )
    original_score_x = score_label.x
    original_score_y = score_label.y

    # Load Images

    title_image             = pyglet.image.load("graphics/undertable_title.png")
    player_image            = pyglet.image.load("graphics/player.png")
    player_jump_image       = pyglet.image.load("graphics/player_jump.png")
    player_shield_image     = pyglet.image.load("graphics/player_shield.png")
    boundary_image          = pyglet.image.load("graphics/boundary.png")
    boss_left_image         = pyglet.image.load("graphics/boss_left.png")
    boss_center_image       = pyglet.image.load("graphics/boss_center.png")
    boss_right_image        = pyglet.image.load("graphics/boss_right.png")
    boss_body_image         = pyglet.image.load("graphics/boss_body.png")
    health_3_image          = pyglet.image.load("graphics/lives_3.png")
    health_2_image          = pyglet.image.load("graphics/lives_2.png")
    health_1_image          = pyglet.image.load("graphics/lives_1.png")
    health_3_jump_image     = pyglet.image.load("graphics/lives_3_jump.png")
    health_2_jump_image     = pyglet.image.load("graphics/lives_2_jump.png")
    health_1_jump_image     = pyglet.image.load("graphics/lives_1_jump.png")
    health_3_shield_image   = pyglet.image.load("graphics/lives_shield_3.png")
    health_2_shield_image   = pyglet.image.load("graphics/lives_shield_2.png")
    health_1_shield_image   = pyglet.image.load("graphics/lives_shield_1.png")
    red_probe_image         = pyglet.image.load("graphics/red_probe.png")
    black_probe_image       = pyglet.image.load("graphics/black_probe.png")
    oscilloscope_image      = pyglet.image.load("graphics/oscilloscope.png")
    large_oscilloscope_image= pyglet.image.load("graphics/large_oscilloscope.png")
    shield_image            = pyglet.image.load("graphics/shield.png")
    pulse_thin_image        = pyglet.image.load("graphics/pulse_thin.png")
    pulse_wide_image        = pyglet.image.load("graphics/pulse_wide.png")
    capacitor_image         = pyglet.image.load("graphics/capacitor.png")

    # Set Anchor Points

    for img in [
        title_image, player_image, player_jump_image, player_shield_image, boundary_image,
        boss_left_image, boss_center_image, boss_right_image, boss_body_image,
        health_3_image, health_2_image, health_1_image,
        health_3_jump_image, health_2_jump_image, health_1_jump_image,
        health_3_shield_image, health_2_shield_image, health_1_shield_image,
        red_probe_image, black_probe_image,
        oscilloscope_image, large_oscilloscope_image, shield_image,
        pulse_thin_image, pulse_wide_image,
        capacitor_image
    ]:
        img.anchor_x = img.width  // 2
        img.anchor_y = img.height // 2

    # Create Sprites

    title_sprite               = pyglet.sprite.Sprite(title_image, x=WIDTH//2, y=HEIGHT//2)
    player_sprite              = pyglet.sprite.Sprite(player_image, x=WIDTH//2, y=HEIGHT//3)
    player_target_x            = player_sprite.x
    player_target_y            = player_sprite.y
    boundary_sprite            = pyglet.sprite.Sprite(boundary_image, x=WIDTH//2, y=HEIGHT//3)
    boss_sprite                = pyglet.sprite.Sprite(boss_center_image, x=WIDTH//2, y=HEIGHT - boss_center_image.height//2)
    oscilloscope_sprite        = pyglet.sprite.Sprite(oscilloscope_image, x=WIDTH*2//3 + 75, y=HEIGHT*4//5 - 25)
    large_oscilloscope_sprite  = pyglet.sprite.Sprite(large_oscilloscope_image, x=WIDTH//2, y=HEIGHT//2)
    boss_body_sprite  = pyglet.sprite.Sprite(boss_body_image, x=WIDTH//2, y=HEIGHT*4//5 - 50)
    boss_body_sprite.opacity = 0

    health_sprite_normal = pyglet.sprite.Sprite(
        health_3_image,
        x=boundary_sprite.x - boundary_sprite.width//2 - health_3_image.width//2 - 10,
        y=boundary_sprite.y
    )
    health_sprite_jump   = pyglet.sprite.Sprite(
        health_3_jump_image,
        x=health_sprite_normal.x, y=health_sprite_normal.y
    )
    health_sprite_shield = pyglet.sprite.Sprite(
        health_3_shield_image,
        x=health_sprite_normal.x, y=health_sprite_normal.y
    )

    shield_sprite = pyglet.sprite.Sprite(shield_image, x=player_sprite.x + SHIELD_RADIUS, y=player_sprite.y)
    shield_sprite.anchor_x = shield_sprite.width  // 2
    shield_sprite.anchor_y = shield_sprite.height // 2

    window.push_handlers(
        on_draw,
        on_mouse_press,
        on_mouse_release,
        key.KeyStateHandler()
    )

    update_config()
    pyglet.clock.schedule_interval(update, 1/30.0)

####################################################################################################
# LOOP                                                                                             #
####################################################################################################

def loop(centroids: DetectedCentroids, dt: float) -> Loop_Result_t:
    global exit_game

    if not mouse_debug:
        if centroids and centroids.fingers:
            centroid = centroids.fingers[0]
            on_mouse_press(centroid.xpos, centroid.ypos, mouse.LEFT, 0)

    pyglet.clock.tick(poll=True)
    window.dispatch_events()
    window.dispatch_event('on_draw')
    window.flip()

    if exit_game or window.has_exit:
        return Loop_Result_t.EXIT
    else:
        return Loop_Result_t.CONTINUE


####################################################################################################
# DEINITIALIZATION                                                                                 #
####################################################################################################

def deinit() -> int:
    global window
    pyglet.clock.unschedule(update)

    end_game()

    reset_game

    if window is not None:
        window.close()
        window = None
    return temp_score

####################################################################################################
# MAIN                                                                                             #
####################################################################################################

def main() -> None:
    cv.cv_init(detect_fingers=True, detect_cds=True)
    init()

    loop_res: Loop_Result_t = Loop_Result_t.CONTINUE
    prev_time = time.time()

    while loop_res == Loop_Result_t.CONTINUE:
        centroids = cv.cv_loop()

        dt = time.time() - prev_time
        prev_time = time.time()

        loop_res = loop(centroids, dt)

    deinit()

if __name__ == "__main__":
    main()
