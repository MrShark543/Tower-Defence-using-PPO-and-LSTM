import pygame
import math
import random
from maps import *
import numpy as np
import copy

# Initialize Pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 40
ENEMY_SPEED = 2
TOWER_RANGE = 150
TOWER_DAMAGE = 10
TOWER_FIRE_RATE = 1  # shots per second
ENEMY_HEALTH = 100
PLAYER_LIVES = 10
PLAYER_MONEY = 800
TOWER_COST = 50
TOWER_UPGRADE_COST = 70

# Enemy types with varying health
ENEMY_TYPES = {
    "normal": {
        "health": 100,
        "speed": 2,
        "color": (255, 0, 0),  # Red
        "reward": 20,
        "damage": 1,  # Lives lost when enemy reaches end
    },
    "fast": {
        "health": 60,
        "speed": 3.5,
        "color": (0, 255, 255),  # Cyan
        "reward": 25,
        "damage": 1,
    },
    "tank": {
        "health": 250,
        "speed": 1.2,
        "color": (100, 100, 100),  # Dark gray
        "reward": 40,
        "damage": 2,
    },
    "boss": {
        "health": 500,
        "speed": 1,
        "color": (128, 0, 128),  # Purple
        "reward": 100,
        "damage": 3,
    }
}

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = [0, 0, 255]
GRAY = (128, 128, 128)
BROWN = (165, 42, 42)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Global display flag
DISPLAY_GAME = False

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tower Defense")

def interpolate_paths(path_dict, step_size=10):
    """
    Interpolates enemy movement paths for smoother animation.

    This function takes a dictionary of paths, where each path is a list of waypoint coordinates.
    It interpolates additional points between each pair of consecutive waypoints based on the 
    specified step size to create smoother movement for enemies along the path.

    Args:
        path_dict (dict): A dictionary where keys are path names (str) and values are lists of 
                          tuples representing (x, y) grid coordinates.
        step_size (int): The maximum pixel distance between interpolated points. Smaller values
                         create more interpolated points for smoother paths. Default is 10.

    Returns:
        dict: A dictionary with the same keys as `path_dict`, where each value is a list of 
              interpolated (x, y) coordinates representing the smooth path.
    """
    interpolated_paths = {}

    for path_name, waypoints in path_dict.items():
        smooth_path = []

        for i in range(len(waypoints)-1):
            start_x, start_y = waypoints[i]
            end_x, end_y = waypoints[i+1]
            distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) 
            steps = max(1, int(distance/step_size))
            smooth_path.append((start_x, start_y))
            for step in range(1, steps):
                x = start_x + (end_x - start_x) * step / steps
                y = start_y + (end_y - start_y) * step / steps
                smooth_path.append((x,y))
        smooth_path.append(waypoints[-1])
        interpolated_paths[path_name] = smooth_path
    return interpolated_paths

def load_map(map_name):
    """
    Loads map data and converts waypoints into interpolated pixel coordinates.

    Retrieves the grid layout and waypoint paths from the specified map name.
    Converts grid-based waypoint coordinates to pixel positions centered within each tile,
    then interpolates those points for smoother enemy movement along the paths.

    Args:
        map_name (str): The key name of the map to load from the `maps` dictionary.

    Returns:
        tuple: A tuple containing:
            - game_map (list of list of int): The 2D grid representing the map layout.
            - all_paths (dict): A dictionary of path names mapped to lists of interpolated pixel (x, y) coordinates.
            - paths (list of str): A list of path names used in the map.
    """
    map_data = maps[map_name]
    game_map = map_data["grid"]

    all_paths = {}
    for path_name, points in map_data["waypoints"].items():
        pixel_points = []
        for grid_x, grid_y in points:
            pixel_x = grid_x * GRID_SIZE + GRID_SIZE // 2
            pixel_y = grid_y * GRID_SIZE + GRID_SIZE // 2
            pixel_points.append((pixel_x, pixel_y))
        all_paths[path_name] = pixel_points
    all_paths = interpolate_paths(all_paths)
    return game_map, all_paths, map_data["paths"]

# Initialize global grid dimensions
GRID_WIDTH = 20
GRID_HEIGHT = 10

font = pygame.font.SysFont(None, 30)

def place_text(text, x, y, color):
    """
    Place text on the screen at specified coordinates."""
    if not DISPLAY_GAME:
        return
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

# Enemy class
class Enemy:
    """
    Represents an enemy unit in the game.

    Args:
        path_name (str): The name of the path this enemy will follow.
        all_paths (dict): A dictionary of all paths available in the game.
        enemy_type (str): The type of enemy (e.g., "normal", "fast", "tank", "boss").
    Attributes:
        path_name (str): The name of the path this enemy will follow.
        path (list): The list of waypoints for this enemy's path.
        path_index (int): The current index in the path.
        position (list): The current position of the enemy in pixel coordinates.
        health (int): The current health of the enemy.
        max_health (int): The maximum health of the enemy.
        speed (int): The speed of the enemy.
        color (tuple): The color of the enemy.
        reward (int): The reward given for defeating this enemy.
        damage (int): The damage this enemy inflicts when it reaches the end of the path.
        enemy_type (str): The type of enemy.
        radius (int): The radius of the enemy for collision detection.
        dead (bool): Whether the enemy is dead.
        reached_end (bool): Whether the enemy has reached the end of its path.

    Methods:
        move(): Moves the enemy along its path.
        take_damage(damage): Reduces the enemy's health by the specified damage amount.
        draw(): Draws the enemy on the screen.

    """
    def __init__(self, path_name, all_paths, enemy_type="normal"):
        self.path_name = path_name
        self.path = all_paths[path_name]
        self.path_index = 0
        self.position = list(self.path[0])
        
        # Load enemy type attributes
        if enemy_type not in ENEMY_TYPES:
            enemy_type = "normal"  # Default to normal if type not found
            
        type_data = ENEMY_TYPES[enemy_type]
        self.health = type_data["health"]
        self.max_health = type_data["health"]
        self.speed = type_data["speed"]
        self.color = type_data["color"]
        self.reward = type_data["reward"]
        self.damage = type_data["damage"]
        
        self.enemy_type = enemy_type
        self.radius = 15
        self.dead = False
        self.reached_end = False
     
    def move(self):
        if self.path_index < len(self.path) - 1:
            target = self.path[self.path_index + 1]
            direction_x = target[0] - self.position[0]
            direction_y = target[1] - self.position[1]
            distance = math.sqrt(direction_x ** 2 + direction_y ** 2)
            
            if distance < self.speed:
                self.path_index += 1
            else:
                self.position[0] += direction_x / distance * self.speed
                self.position[1] += direction_y / distance * self.speed
        else:
            # Enemy reached the end of the path
            self.reached_end = True
            self.dead = True
    
    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.dead = True
    
    def draw(self):
        if not DISPLAY_GAME:
            return
            
        # Draw enemy
        pygame.draw.circle(screen, self.color, (int(self.position[0]), int(self.position[1])), self.radius)
        
        # Draw health bar
        health_bar_length = 30
        health_bar_height = 5
        health_ratio = self.health / self.max_health
        health_bar_fill = health_ratio * health_bar_length
        
        pygame.draw.rect(screen, BLACK, (self.position[0] - health_bar_length // 2, 
                                    self.position[1] - self.radius - 10, 
                                    health_bar_length, health_bar_height))
        pygame.draw.rect(screen, GREEN, (self.position[0] - health_bar_length // 2, 
                                    self.position[1] - self.radius - 10, 
                                    health_bar_fill, health_bar_height))

# Tower class
class Tower:
    """
    Represents a tower in the game.
    
    Args:
        x (int): The x-coordinate of the tower's position.
        y (int): The y-coordinate of the tower's position.
    Attributes:
        position (tuple): The (x, y) coordinates of the tower.
        grid_position (tuple): The grid coordinates of the tower.
        range (int): The attack range of the tower.
        damage (int): The damage dealt by the tower.
        fire_rate (float): The rate at which the tower fires.
        last_shot_time (int): The last time the tower fired a shot.
        target (Enemy): The current target enemy.
        upgrade_multiplier (int): The multiplier for upgrade costs.
        color (list): The color of the tower.
        level (int): The current level of the tower.
        shots_fired (int): The total number of shots fired by the tower.
        shots_fired_this_wave (int): The number of shots fired in the current wave.
        tower_rect (pygame.Rect): The rectangle representing the tower's position and size.
    Methods:
        find_target(enemies): Finds the closest target enemy within range.
        shoot(frame_counter): Fires at the target enemy if the fire rate allows.
        reset_wave_shots(): Resets the shots fired in the current wave.
        draw(): Draws the tower and its target line on the screen.
        upgrade(): Upgrades the tower's stats.


    """
    def __init__(self, x, y):
        self.position = (x, y)
        self.grid_position = (x // GRID_SIZE, y // GRID_SIZE)
        self.range = TOWER_RANGE
        self.damage = TOWER_DAMAGE
        self.fire_rate = TOWER_FIRE_RATE
        self.last_shot_time = 0  # controls fire rate
        self.target = None
        self.upgrade_multiplier = 1
        self.color = [0, 0, 255]
        self.level = 1
        self.shots_fired = 0  # Track total shots fired
        self.shots_fired_this_wave = 0  # Track shots fired in current wave
        self.tower_rect = pygame.Rect(
            self.position[0] - GRID_SIZE // 2,
            self.position[1] - GRID_SIZE // 2,
            GRID_SIZE,
            GRID_SIZE
        )
    
    def find_target(self, enemies):
        self.target = None
        shortest_distance = float('inf')
        
        for enemy in enemies:
            if enemy.dead:
                continue
            
            distance = math.sqrt((enemy.position[0] - self.position[0]) ** 2 + 
                                 (enemy.position[1] - self.position[1]) ** 2)
            if distance < self.range and distance < shortest_distance:
                shortest_distance = distance
                self.target = enemy
    
    def shoot(self, frame_counter):
        frames_per_shot = int(30 / self.fire_rate)
        if self.target and (frame_counter - self.last_shot_time) >= frames_per_shot:
            self.target.take_damage(self.damage)
            self.last_shot_time = frame_counter
            self.shots_fired += 1  # Increment total shots counter
            self.shots_fired_this_wave += 1  
            return True
        return False
    
    def reset_wave_shots(self):
        self.shots_fired_this_wave = 0
    
    def draw(self):
        if not DISPLAY_GAME:
            return
            
        # Draw tower
        self.tower_rect = pygame.draw.rect(screen, self.color, 
                                        (self.position[0] - GRID_SIZE // 2, 
                                        self.position[1] - GRID_SIZE // 2, 
                                        GRID_SIZE, GRID_SIZE))
        place_text(f"{self.level}", (self.position[0] - GRID_SIZE // 2) + 15, (self.position[1] - GRID_SIZE // 2) + 10, WHITE)
        
        # Draw line to target
        if self.target and not self.target.dead:
            pygame.draw.line(screen, self.color, self.position, 
                        (self.target.position[0], self.target.position[1]), 2)
   
    def upgrade(self):
        if self.level <= 4:
            self.damage += 5  # Increase damage
            self.fire_rate += 0.2  # Increase fire rate
            self.upgrade_multiplier += 2
            self.color[2] -= 25 if self.color[2] > 0 else 0
            self.color[0] += 25 if self.color[0] < 255 else 0
            self.level += 1

# Projectile class
class Projectile:
    """
    Represents a projectile fired by a tower.

    Args:
        start_pos (tuple): The starting position of the projectile.
        target (Enemy): The target enemy the projectile is aimed at.

    Attributes:
        position (list): The current position of the projectile.
        target (Enemy): The target enemy.
        speed (int): The speed of the projectile.
        radius (int): The radius of the projectile.
        reached_target (bool): Whether the projectile has reached its target.
    """
    def __init__(self, start_pos, target):
        self.position = list(start_pos)
        self.target = target
        self.speed = 10
        self.radius = 5
        self.reached_target = False
    
    def move(self):
        if self.target.dead:
            self.reached_target = True
            return
        
        direction_x = self.target.position[0] - self.position[0]
        direction_y = self.target.position[1] - self.position[1]
        distance = math.sqrt(direction_x ** 2 + direction_y ** 2)
        
        if distance < self.speed:
            self.reached_target = True
        else:
            self.position[0] += direction_x / distance * self.speed
            self.position[1] += direction_y / distance * self.speed
    
    def draw(self):
        if not DISPLAY_GAME:
            return
            
        pygame.draw.circle(screen, BLUE, (int(self.position[0]), int(self.position[1])), self.radius)

# Wave Manager class
class WaveManager:
    """
    Manages the spawning of enemies in waves.

    Args:
        paths (list): A list of path names where enemies will spawn.
        max_waves (int): The maximum number of waves to spawn.
        wave_compositions (list): A list of dictionaries defining the composition of each wave.
    
    Attributes:
        current_wave (int): The current wave number.
        enemy_count (int): The total number of enemies in the current wave.
        enemies_spawned (int): The number of enemies spawned so far in the current wave.
        time_between_spawns (int): The time between enemy spawns in frames.
        last_spawn_time (int): The last time an enemy was spawned.
        wave_cleared (bool): Whether the current wave has been cleared.
        max_waves (int): The maximum number of waves.
        paths (list): A list of path names where enemies will spawn.
        current_composition (dict): The current wave composition.
        wave_compositions (list): A list of dictionaries defining the composition of each wave.
    Methods:
        start_next_wave(): Starts the next wave of enemies.
        get_next_enemy_type(): Returns the next enemy type to spawn based on the current wave composition.
        spawn_enemy(frame_counter, enemies_list, all_paths): Spawns an enemy if conditions are met.

    
    """
    def __init__(self, paths, max_waves=3, wave_compositions=None):
        self.current_wave = 0
        self.enemy_count = 0
        self.enemies_spawned = 0
        self.time_between_spawns = 30  # in frames (30 frames = 1 second at 30 FPS)
        self.last_spawn_time = 0
        self.wave_cleared = False
        self.max_waves = max_waves
        self.paths = paths
        self.current_composition = {}  # Initialize empty current composition
        
        if wave_compositions is not None:
            # print(f"Using provided wave compositions with {len(wave_compositions)} waves")
            self.wave_compositions = wave_compositions
        else:
            # print("No wave compositions provided, generating defaults")
            self.wave_compositions = self._generate_default_wave_compositions()
            # print(f"Generated {len(self.wave_compositions)} default wave compositions")
    
    def _generate_default_wave_compositions(self):
        """Generate default wave compositions if none provided from map"""
        default_compositions = [
            # Wave 1
            {"normal": 10, "fast": 0, "tank": 0, "boss": 0},
            # Wave 2
            {"normal": 8, "fast": 5, "tank": 0, "boss": 0},
            # Wave 3
            {"normal": 10, "fast": 8, "tank": 2, "boss": 0},
        ]
        return default_compositions
    
    def start_next_wave(self):
        if self.current_wave < self.max_waves and self.current_wave < len(self.wave_compositions):
            self.current_wave += 1
            
            wave_idx = min(self.current_wave - 1, len(self.wave_compositions) - 1)
            
            
            composition = copy.deepcopy(self.wave_compositions[wave_idx])
            
           
            # print(f"Starting wave {self.current_wave} with composition: {composition}")
            
            if not composition or sum(composition.values()) == 0:
                # print("WARNING: Wave composition has no enemies, using fallback")
                composition = {"normal": 10, "fast": 5, "tank": 2, "boss": 0}
            
            self.enemy_count = sum(composition.values())
            # print(f"Total enemies for this wave: {self.enemy_count}")
            
            # Store the composition for this wave separately
            self.current_composition = composition
            
            self.enemies_spawned = 0
            self.wave_cleared = False
            self.last_spawn_time = 0
            return True
        
        # print(f"Cannot start next wave: current_wave={self.current_wave}, max_waves={self.max_waves}, available compositions={len(self.wave_compositions)}")
        return False
    
    def get_next_enemy_type(self):
        """Get the next enemy type to spawn based on wave composition"""
        if self.current_wave == 0 or self.enemies_spawned >= self.enemy_count:
            return None
        
        if not self.current_composition:
            # print("ERROR: No current composition found. Using fallback.")
            self.current_composition = {"normal": 10, "fast": 0, "tank": 0, "boss": 0}
        
        # Create a weighted list based on remaining enemies
        weighted_types = []
        for enemy_type, count in self.current_composition.items():
            if count > 0:  # Only include enemy types with positive counts
                weighted_types.extend([enemy_type] * count)
        
        if not weighted_types:
            # print("No enemy types left in wave composition")
            return None
        
        # Select a random enemy type from the weighted list
        enemy_type = random.choice(weighted_types)
        
        # Decrement the count for this enemy type
        self.current_composition[enemy_type] -= 1
        
        return enemy_type
    
    def spawn_enemy(self, frame_counter, enemies_list, all_paths):
        if self.enemies_spawned >= self.enemy_count:
            if not enemies_list:  # If no enemies are left
                self.wave_cleared = True
            return False
        
        # if frame_counter % 60 == 0:  
        #     print(f"Attempting to spawn enemy. Spawned: {self.enemies_spawned}/{self.enemy_count}")
        
        if frame_counter - self.last_spawn_time >= self.time_between_spawns:
            enemy_type = self.get_next_enemy_type()
            
            if enemy_type:
                # Choose a random path from available paths
                if not self.paths:
                    # print("ERROR: No paths available. Using fallback.")
                    path_choice = "path1"  # Fallback
                else:
                    path_choice = random.choice(self.paths)
                
                # Create enemy
                try:
                    new_enemy = Enemy(path_choice, all_paths, enemy_type)
                    enemies_list.append(new_enemy)
                    
                    self.enemies_spawned += 1
                    self.last_spawn_time = frame_counter
                    # print(f"Spawned {enemy_type} enemy on {path_choice}. Total: {self.enemies_spawned}/{self.enemy_count}")
                    return True
                except Exception as e:
                    print(f"ERROR spawning enemy: {e}")
            
        return False


class GameManager:
    """
Manages the entire game state, rendering, game logic, wave handling, and AI integration.

The GameManager coordinates all components of the game including enemies, towers, wave progression,
economy, reward computation, and state transitions. It also supports reinforcement learning through
step-based game progression and reward feedback.

Attributes:
        enemies (list): List of currently active enemies in the game.
        towers (list): List of all towers placed by the player.
        projectiles (list): List of active projectiles in the game.
        player_lives (int): Number of lives the player has remaining.
        player_money (int): Amount of in-game currency the player currently has.
        score (int): Player’s accumulated score.
        game_over (bool): Flag indicating if the game has ended.
        game_map (list): 2D grid representing the current map layout.
        all_paths (dict): Interpolated enemy paths for each path name.
        paths (list): List of path names used in the current map.
        frame_counter (int): Total number of frames since the game started.
        wave_manager (WaveManager): Manages enemy waves and spawns.
        wave_number (int): The current wave number.
        reward (float): Reward value calculated at each step for RL agents.
        enemies_killed (int): Total number of enemies eliminated.
        enemies_leaked (int): Total number of enemies that reached the end of the path.
        current_map (str): Identifier for the current map being used.
        steps_taken (int): Total number of time steps taken since game start.

Methods:
        initialize_wave_manager(): Initializes the WaveManager with map-specific data.
        load_wave_data(): Loads wave composition and wave count from the current map data.
        draw_grid(): Draws the background grid on the screen.
        draw_map(): Renders the map layout based on tile types.
        is_buildable(x, y): Checks if a grid cell can have a tower built on it.
        is_upgradable(x, y): Attempts to upgrade a tower at the specified pixel position.
        can_afford_tower(): Returns True if the player can afford a new tower.
        can_upgrade_tower(tower): Returns True if a given tower can be upgraded.
        can_be_upgraded(tower_position): Returns True if a tower at a given pixel position is upgradable.
        draw_ui(): Renders UI elements like lives, money, score, and wave info.
        calculate_path_coverage(): Calculates the percentage of path tiles covered by tower range.
        draw_game_entities(): Renders all in-game entities (towers, enemies, projectiles).
        reset_game(): Resets the game to its initial state, including all entities and counters.
        change_map(new_map_name): Loads and switches to a new map, resetting the game state.
        update(): Updates game logic for a single frame and returns the reward and end condition.
        place_tower(grid_x, grid_y): Places a tower at a given grid coordinate if allowed.
        display(): Renders the current game frame if display is enabled.
        get_state(): Returns the current state representation for reinforcement learning agents.
        handle_wave_transition(): Handles the end of wave events and starts the next wave.
        calculate_tower_firing_reward(): Computes a reward based on tower performance in a wave.
        get_strategic_positions(): Returns categorized tower placement positions by strategic distance.
        step(action): Performs a single game step using a discrete action.
        get_performance_metrics(): Returns a dictionary of game performance metrics.

"""
    def __init__(self):
        # Game states
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.spawn_timer = 0
        self.player_lives = PLAYER_LIVES
        self.player_money = PLAYER_MONEY
        self.game_over = False
        self.score = 0
        
        # Set default map and load it
        self.current_map = "map1_2"
        self.game_map, self.all_paths, self.paths = load_map(self.current_map)
        
        # Initialize timing and dimensions
        self.frame_counter = 0
        self.clock = pygame.time.Clock()
        self.GRID_WIDTH = len(self.game_map[0])
        self.GRID_HEIGHT = len(self.game_map)
        self.current_wave_started = False
        self.wave_start_time = 0
        
        # Set default values
        self.wave_number = 0  # Start at wave 0, increments to 1 on first wave
        self.enemies_spawned = 0
        self.wave_cleared = False
        self.steps_taken = 0
        self.reward = 0
        
        # Set default max waves
        self.max_waves = 3
        
        # Get wave data from map if available
        self.load_wave_data()
        
        # Initialize wave manager
        self.initialize_wave_manager()
        
        # Update global grid dimensions
        global GRID_HEIGHT, GRID_WIDTH
        GRID_WIDTH = self.GRID_WIDTH
        GRID_HEIGHT = self.GRID_HEIGHT
        
        # Enemy stats tracking
        self.enemies_killed = 0
        self.enemies_leaked = 0
    
    def initialize_wave_manager(self):
       
        # print(f"Initializing wave manager with paths: {self.paths}, max_waves: {self.max_waves}")
        
        
        wave_comps = None
        if hasattr(self, 'wave_compositions') and self.wave_compositions:
            import copy
            wave_comps = copy.deepcopy(self.wave_compositions)
            # print(f"Using custom wave compositions with {len(wave_comps)} waves")
        else:
            print("No custom wave compositions, using defaults")
        
        # Create the wave manager
        self.wave_manager = WaveManager(self.paths, self.max_waves, wave_comps)
        
        # Start the first wave
        success = self.wave_manager.start_next_wave()
        if success:
            self.wave_number = self.wave_manager.current_wave
            print(f"Started wave {self.wave_number}")
        else:
            print("Failed to start first wave!")
    
    def load_wave_data(self):
        try:
            map_data = maps[self.current_map]
            # print(f"Loading wave data for map: {self.current_map}")
            
           
            if "max_waves" in map_data:
                self.max_waves = map_data["max_waves"]
                print(f"Max waves from map data: {self.max_waves}")
            # else:
           
            #     print(f"No max_waves in map data, using default: {self.max_waves}")
                    
            if "wave_compositions" in map_data:
                import copy
                self.wave_compositions = copy.deepcopy(map_data["wave_compositions"])
                # print(f"Wave compositions loaded: {len(self.wave_compositions)} waves")
                # for i, wave in enumerate(self.wave_compositions):
                #     print(f"Wave {i+1}: {wave}")
            else:
                self.wave_compositions = None  # Will use defaults
                print("No wave compositions found in map data, will use defaults")
        except Exception as e:
            print(f"ERROR loading wave data: {e}")
            # Set fallback values
            self.max_waves = 3
            self.wave_compositions = None
    
    def draw_grid(self):
        if not DISPLAY_GAME:
            return
            
        try:
            for x in range(0, SCREEN_WIDTH, GRID_SIZE):
                pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_HEIGHT), 1)
            for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
                pygame.draw.line(screen, GRAY, (0, y), (SCREEN_WIDTH, y), 1)
        except pygame.error as e:
            print(f"Warning: Failed to draw grid: {e}")

    def draw_map(self):
        if not DISPLAY_GAME:
            return
            
        try:
            for y, row in enumerate(self.game_map):
                for x, cell in enumerate(row):
                    rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                    if cell in [1, 3]:  # Path tiles
                        pygame.draw.rect(screen, BROWN, rect)
                    elif cell == 5:  # Converged path
                        pygame.draw.rect(screen, BROWN, rect)
                    elif cell == 2:  # Tower placable tile
                        pygame.draw.rect(screen, GRAY, rect)
                    else:  # grass
                        pygame.draw.rect(screen, GREEN, rect)
        except pygame.error as e:
            print(f"Warning: Failed to draw map: {e}")

    def is_buildable(self, x, y):
        grid_x = x // GRID_SIZE
        grid_y = y // GRID_SIZE
        if grid_x < 0 or grid_x >= len(self.game_map[0]) or grid_y < 0 or grid_y >= len(self.game_map):
            return False
        return self.game_map[grid_y][grid_x] == 2 and not any(
            tower.grid_position == (grid_x, grid_y) for tower in self.towers
        )

    def is_upgradable(self, x, y):
        for tower in self.towers:
            if tower.tower_rect.collidepoint((x, y)):
                if self.can_upgrade_tower(tower):
                    self.player_money -= TOWER_UPGRADE_COST * tower.upgrade_multiplier
                    tower.upgrade()
                    return True
        return False

    def can_afford_tower(self):
        return self.player_money >= TOWER_COST

    def can_upgrade_tower(self, tower):
        return self.player_money >= TOWER_UPGRADE_COST * tower.upgrade_multiplier
    
    def can_be_upgraded(self, tower_position):
        for tower in self.towers:
            if tower.tower_rect.collidepoint(tower_position):
                return self.can_upgrade_tower(tower)
        return False

    def draw_ui(self):
        if not DISPLAY_GAME:
            return
            
        # Draw lives
        place_text(f"Lives: {self.player_lives}", 10, 10, BLACK)
        
        # Draw money
        place_text(f"Money: ${self.player_money}", 10, 40, BLACK)
        
        # Draw score
        place_text(f"Score: {self.score}", 10, 70, BLACK)
        
        # Draw wave number
        place_text(f"Wave: {self.wave_number}/{self.wave_manager.max_waves}", 10, 100, BLACK)
        
        # Draw current map
        place_text(f"Map: {self.current_map}", 10, 130, BLACK)

        # Game over message
        if self.game_over:
            place_text("GAME OVER!!", SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2, RED)
        
        # Wave cleared message 
        elif self.wave_manager.wave_cleared and not self.enemies:
            place_text("Wave Cleared!!", SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2, PURPLE)
    
    def calculate_path_coverage(self):
        path_cells = [(x, y) for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH) 
                    if self.game_map[y][x] in [1, 3, 5]]
        
        if not path_cells:
            return 0.0
        
        covered_cells = set()
        for tower in self.towers:
            for px, py in path_cells:
                path_pos = (px * GRID_SIZE + GRID_SIZE // 2, py * GRID_SIZE + GRID_SIZE // 2)
                dist = math.sqrt((tower.position[0] - path_pos[0])**2 + 
                            (tower.position[1] - path_pos[1])**2)
                if dist <= tower.range:
                    covered_cells.add((px, py))
        
        return len(covered_cells) / len(path_cells)
    
    def draw_game_entities(self):
        if not DISPLAY_GAME:
            return
            
        try:
            for tower in self.towers:
                tower.draw()

            for projectile in self.projectiles:
                projectile.draw()

            for enemy in self.enemies:
                enemy.draw()
            
            self.draw_ui()
        except (TypeError, pygame.error) as e:
            print(f"Warning: Failed to draw game entities: {e}")
        
    def reset_game(self):
        """Reset the game to initial state"""
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.player_lives = PLAYER_LIVES
        self.player_money = PLAYER_MONEY
        self.game_over = False
        self.score = 0
        self.wave_number = 0
        self.enemies_spawned = 0
        self.wave_cleared = False
        self.reward = 0
        self.steps_taken = 0
        self.current_wave_started = False
        self.wave_start_time = 0
        self.enemies_killed = 0
        self.enemies_leaked = 0
        
        self.initialize_wave_manager()
    
    def change_map(self, new_map_name):
        """Change to a new map"""
        self.current_map = new_map_name
        self.game_map, self.all_paths, self.paths = load_map(self.current_map)
        
        # Load wave data for the new map
        self.load_wave_data()
        
        # Reset game with new map data
        self.reset_game()
        
        # Update grid dimensions
        self.GRID_WIDTH = len(self.game_map[0])
        self.GRID_HEIGHT = len(self.game_map)
        global GRID_HEIGHT, GRID_WIDTH
        GRID_WIDTH = self.GRID_WIDTH
        GRID_HEIGHT = self.GRID_HEIGHT
    
    def update(self):
        """Update game state for one frame"""
        step_reward = 0
        enemies_killed = 0
        
        # Small per-step penalty to avoid large negative rewards over time
        step_reward -= 0.0001
        
        self.frame_counter += 1
        
        # Check for wave transitions and get firing reward
        wave_transition_reward = self.handle_wave_transition()
        step_reward += wave_transition_reward
        
        if not self.game_over and not self.wave_manager.wave_cleared:
            # Debug wave manager state (but not every frame to reduce spam)
            # if self.frame_counter % 60 == 0:
            #     print(f"Wave: {self.wave_number}, Enemies spawned: {self.wave_manager.enemies_spawned}/{self.wave_manager.enemy_count}")
            
            # Spawn enemies using WaveManager
            spawned = self.wave_manager.spawn_enemy(self.frame_counter, self.enemies, self.all_paths)
            
            # Check if wave is cleared
            if self.wave_manager.wave_cleared and not self.enemies:
                # print("Wave cleared!")
                step_reward += 150
                self.player_money += 100
            
            # Update towers (targeting and shooting)
            for tower in self.towers:
                tower.find_target(self.enemies)
                if tower.shoot(self.frame_counter):
                    self.projectiles.append(Projectile(tower.position, tower.target))
                    # Small reward for shooting
                    step_reward += 0.2
            
            # Update projectiles
            for projectile in self.projectiles[:]:
                projectile.move()
                if projectile.reached_target:
                    self.projectiles.remove(projectile)
            
            # Update enemies
            for enemy in self.enemies[:]:
                enemy.move()
                
                if enemy.reached_end:
                    self.player_lives -= enemy.damage
                    step_reward -= 5 * enemy.damage  # Scale penalty by damage
                    self.enemies_leaked += 1
                    if self.player_lives <= 0:
                        self.game_over = True
                        step_reward -= 50
                
                if enemy.dead:
                    if not enemy.reached_end:
                        self.player_money += enemy.reward
                        self.score += enemy.reward // 2
                        enemies_killed += 1
                        self.enemies_killed += 1
                        enemy_kill_reward = enemy.reward * 0.75
                        step_reward += enemy_kill_reward
                    self.enemies.remove(enemy)
        
    
        # Add strategic rewards for general principles
        coverage_percentage = self.calculate_path_coverage()
        if coverage_percentage > 0.7:  # If covering more than 70% of the path
            step_reward += 0.5  # Small constant reward for good coverage
        
        step_reward += 0.005  # small reward per step
        self.steps_taken += 1
        self.reward = step_reward
        # Increase wave clear bonus based on wave number
        if self.wave_manager.wave_cleared and not self.enemies:
            step_reward += 250 + (self.wave_number * 30)
        
        # Reward for having towers placed 
        step_reward += len(self.towers) * 0.1
        
        # Additional reward for tower upgrades
        upgraded_tower_count = sum(1 for tower in self.towers if tower.level > 1)
        step_reward += upgraded_tower_count * 0.2
        
        step_reward+=0.005 #small reward per step
        self.steps_taken += 1
        self.reward = step_reward
        return step_reward, self.game_over or self.wave_manager.wave_cleared

    def place_tower(self, grid_x, grid_y):
        """Place a tower at the specified grid position"""
        pixel_x = grid_x * GRID_SIZE + GRID_SIZE // 2
        pixel_y = grid_y * GRID_SIZE + GRID_SIZE // 2
        
        if self.can_afford_tower() and not any(tower.grid_position == (grid_x, grid_y) for tower in self.towers):
            self.towers.append(Tower(pixel_x, pixel_y))
            self.player_money -= TOWER_COST
            return True
        return False

    def display(self):
        """Render the game to the screen"""
        if not DISPLAY_GAME:
            return
            
        screen.fill(WHITE)
        
        self.draw_map()
        self.draw_grid()
        self.draw_game_entities()
        
    
        pygame.display.update()
   
    def get_state(self):
        """
        Create a state representation using relative distances from paths.
        """
        # Distance maps 
        path_distance_map = np.ones((GRID_HEIGHT, GRID_WIDTH)) * 999

        path_density = sum(1 for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH) 
                      if self.game_map[y][x] in [1, 3, 5]) / (self.GRID_HEIGHT * self.GRID_WIDTH)
        

        # Get path characteristics
        path_segments = []
        for path_name, waypoints in self.all_paths.items():
            path_length = sum(math.sqrt((waypoints[i+1][0] - waypoints[i][0])**2 + 
                                    (waypoints[i+1][1] - waypoints[i][1])**2) 
                            for i in range(len(waypoints)-1))
            path_segments.append(path_length)
        
        avg_path_length = sum(path_segments) / len(path_segments) if path_segments else 0
        path_length_normalized = avg_path_length / (self.GRID_WIDTH * GRID_SIZE)
        
        # Find all path cells
        path_cells = []
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game_map[y][x] in [1, 3, 5]:  # Path tiles
                    path_cells.append((x, y))
        
        # Calculate distance from each cell to nearest path
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                for px, py in path_cells:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    path_distance_map[y][x] = min(path_distance_map[y][x], dist)
        
        # Normalize distances
        max_dist = np.max(path_distance_map)
        if max_dist > 0:  # Avoid division by zero
            path_distance_map = path_distance_map / max_dist
        
        # Existing tower placement grid
        tower_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for tower in self.towers:
            x, y = tower.grid_position
            if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                tower_grid[y][x] = 1
        
        # Enemy density map
        enemy_grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for enemy in self.enemies:
            try:
                grid_x = int(enemy.position[0] // GRID_SIZE)
                grid_y = int(enemy.position[1] // GRID_SIZE)
                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                    enemy_grid[grid_y][grid_x] += enemy.health / enemy.max_health
            except (AttributeError, IndexError, TypeError) as e:
                print(f"Error updating enemy grid: {e}")
        
        # Path coverage 
        coverage_map = np.zeros((GRID_HEIGHT, GRID_WIDTH))
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game_map[y][x] == 2:  
                    tower_pos = (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2)
                    coverage = 0
                    for px, py in path_cells:
                        path_pos = (px * GRID_SIZE + GRID_SIZE // 2, py * GRID_SIZE + GRID_SIZE // 2)
                        dist = math.sqrt((tower_pos[0] - path_pos[0])**2 + (tower_pos[1] - path_pos[1])**2)
                        if dist <= TOWER_RANGE:
                            coverage += 1
                    if path_cells:  # Avoid division by zero
                        coverage_map[y][x] = coverage / len(path_cells)
        
        state = np.stack([
            tower_grid,
            path_distance_map,
            coverage_map,
            enemy_grid
        ], axis=0)
    
        # Scalar state
        scalar_state = np.array([
            self.player_money / 500,
            self.wave_number / max(1, self.wave_manager.max_waves),  # Avoid division by zero
            len(self.enemies) / 20,
            self.player_lives / PLAYER_LIVES,
            path_density,
            path_length_normalized
        ])
        
        return {
            'grid_state': state,
            'scalar_state': scalar_state
        }
    
    def handle_wave_transition(self):
        """Handle wave transitions and return rewards"""
        # If a new wave just started
        if not self.current_wave_started and self.wave_manager.enemies_spawned > 0:
            self.current_wave_started = True
            self.wave_start_time = self.frame_counter
            # Reset shot counters for all towers
            for tower in self.towers:
                tower.reset_wave_shots()
        
        # If a wave just ended
        if self.wave_manager.wave_cleared and not self.enemies:
            # Calculate firing activity reward
            firing_reward = self.calculate_tower_firing_reward()
            
            # Check if all waves completed
            if self.wave_number < self.wave_manager.max_waves:
                # Start next wave
                if self.wave_manager.start_next_wave():
                    self.wave_number = self.wave_manager.current_wave
                    self.current_wave_started = False
                    
            return firing_reward
        
        return 0

    def calculate_tower_firing_reward(self):
        """Calculate reward based on tower firing activity, with penalties for inactive towers"""
        if not self.towers:
            return 0
        
        total_reward = 0
        wave_duration = max(1, self.frame_counter - self.wave_start_time)  # Avoid division by zero
        
        # Track towers that haven't fired or fired too little
        inactive_towers = []
        underperforming_towers = []
        
        # Calculate total shots and identify inactive/underperforming towers
        total_shots = 0
        for tower in self.towers:
            shots = tower.shots_fired_this_wave
            total_shots += shots
            
            # Expected shots based on tower's fire rate and wave duration
            expected_shots = (tower.fire_rate * wave_duration) / 30
            
            # Identify inactive towers (no shots at all)
            if shots == 0:
                inactive_towers.append(tower)
            # Identify underperforming towers (less than 30% of expected shots)
            elif shots < expected_shots * 0.3:
                underperforming_towers.append(tower)
        
        # Reward based on average shot frequency
        shot_frequency = total_shots / (wave_duration / 30)  # shots per second
        frequency_reward = min(50, shot_frequency * 2)  # Cap at 50 to prevent excessive reward
        
        # Extra reward for well-distributed firing (all towers active)
        active_towers = sum(1 for tower in self.towers if tower.shots_fired_this_wave > 0)
        active_tower_ratio = active_towers / len(self.towers) if self.towers else 0
        distribution_reward = active_tower_ratio * 20  # Up to 20 reward points for all towers being active
        
        # Calculate total positive reward
        positive_reward = frequency_reward + distribution_reward
        
        # Scaling penalties based on tower level and investment
        scaled_inactive_penalty = 0
        for tower in inactive_towers:
            # Get tower position
            grid_x, grid_y = tower.grid_position
            
            # Find distance to nearest path
            min_dist = float('inf')
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if self.game_map[y][x] in [1, 3, 5]:  # Path tiles
                        dist = math.sqrt((grid_x - x)**2 + (grid_y - y)**2)
                        min_dist = min(min_dist, dist)
            
            # Higher penalty for strategic positions that aren't being utilized
            position_multiplier = 1.0
            if min_dist < 1.5:  # Very close to path
                position_multiplier = 1.5
            
            # Higher level towers have higher penalties since they cost more
            scaled_inactive_penalty += 2 * tower.level * position_multiplier
        
        scaled_underperforming_penalty = 0
        for tower in underperforming_towers:
            scaled_underperforming_penalty += 2 * tower.level
        
        # Calculate final reward (positive rewards minus penalties)
        total_reward = positive_reward - (scaled_inactive_penalty + scaled_underperforming_penalty)
        
        # Print debug info if rendering
        if DISPLAY_GAME:
            print(f"Wave {self.wave_number} firing stats:")
            print(f"  Total shots: {total_shots}")
            print(f"  Avg shots per tower: {total_shots / len(self.towers):.1f}")
            print(f"  Shot frequency: {shot_frequency:.1f} shots/sec")
            print(f"  Active towers: {active_towers}/{len(self.towers)} ({active_tower_ratio:.0%})")
            print(f"  Inactive towers: {len(inactive_towers)}")
            print(f"  Underperforming towers: {len(underperforming_towers)}")
            print(f"  Positive reward: {positive_reward:.1f}")
            print(f"  Inactive penalty: {scaled_inactive_penalty:.1f}")
            print(f"  Underperforming penalty: {scaled_underperforming_penalty:.1f}")
            print(f"  Total reward: {total_reward:.1f}")
        
        return total_reward
    
    def get_strategic_positions(self):
        """Get positions based on strategic value rather than fixed coordinates"""
        # Find all buildable positions
        buildable = []
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.game_map[y][x] == 2:
                    buildable.append((x, y))
        
        # Group positions by strategic value
        close_to_path = []
        medium_distance = []
        far_from_path = []
        
        # Calculate path coverage for each position
        path_cells = [(x, y) for y in range(GRID_HEIGHT) for x in range(GRID_WIDTH) 
                    if self.game_map[y][x] in [1, 3, 5]]
        
        for x, y in buildable:
            tower_pos = (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2)
            
            # Find nearest path cell
            min_dist = float('inf')
            for px, py in path_cells:
                path_pos = (px * GRID_SIZE + GRID_SIZE // 2, py * GRID_SIZE + GRID_SIZE // 2)
                dist = math.sqrt((tower_pos[0] - path_pos[0])**2 + (tower_pos[1] - path_pos[1])**2)
                min_dist = min(min_dist, dist)
            
            # Categorize by distance
            if min_dist < GRID_SIZE * 1.5:
                close_to_path.append((x, y))
            elif min_dist < GRID_SIZE * 3:
                medium_distance.append((x, y))
            else:
                far_from_path.append((x, y))
        
        # Return all categories
        return {
            'close': close_to_path,
            'medium': medium_distance,
            'far': far_from_path
        }

    def step(self, action):
        """
        Take a step using strategic positions for better generalization across maps.
        
        Actions:
        - 0-9: Place tower at 'close' position (0-9)
        - 10-19: Place tower at 'medium' position (0-9)
        - 20-29: Place tower at 'far' position (0-9)
        - 30-39: Upgrade tower at 'close' position (0-9)
        - 40-49: Upgrade tower at 'medium' position (0-9)
        - 50-59: Upgrade tower at 'far' position (0-9)
        - 60: Do nothing (save money)
        """
        strategic_positions = self.get_strategic_positions()
        action_reward = 0
        max_positions_per_category = 10  # Maximum positions we consider per category
        
        # Decode the action
        if action < 3 * max_positions_per_category:  # Tower placement actions
            # Determine category
            category_index = action // max_positions_per_category
            position_index = action % max_positions_per_category
            
            # Map category index to category name
            categories = ['close', 'medium', 'far']
            category = categories[category_index]
            
            # Get positions for this category
            positions = strategic_positions[category]
            
            # Check if position exists
            if position_index < len(positions):
                grid_x, grid_y = positions[position_index]
                placed = self.place_tower(grid_x, grid_y)
                if placed:
                    # Extra reward for placing in more strategic locations
                    base_reward = 5
                    if category == 'close':
                        action_reward += base_reward * 1.5  # More reward for close positions
                    elif category == 'medium':
                        action_reward += base_reward * 1.0
                    else:  # 'far'
                        action_reward += base_reward * 0.7  # Less reward for far positions
        
        elif action < 6 * max_positions_per_category:  # Tower upgrade actions
            adjusted_action = action - (3 * max_positions_per_category)
            
            # Determine category for upgrade
            category_index = adjusted_action // max_positions_per_category
            position_index = adjusted_action % max_positions_per_category
            
            # Map category index to category name
            categories = ['close', 'medium', 'far']
            category = categories[category_index]
            
            # Get positions for this category
            positions = strategic_positions[category]
            
            # Check if position exists
            if position_index < len(positions):
                grid_x, grid_y = positions[position_index]
                pos_x = grid_x * GRID_SIZE + GRID_SIZE // 2
                pos_y = grid_y * GRID_SIZE + GRID_SIZE // 2
                upgraded = self.is_upgradable(pos_x, pos_y)
                if upgraded:
                    # Extra reward for upgrading more strategic locations
                    base_reward = 10
                    if category == 'close':
                        action_reward += base_reward * 1.5
                    elif category == 'medium':
                        action_reward += base_reward * 1.0
                    else:  # 'far'
                        action_reward += base_reward * 0.7
        
        # else: do nothing (save money)
        
        # Update game state
        game_reward, done = self.update()
        
        # Combine action reward with game reward
        reward = action_reward + game_reward
        
        if self.wave_manager.wave_cleared and not self.enemies:
            if self.wave_number >= self.wave_manager.max_waves:
                reward += 500  # Big bonus for completing all waves
                done = True
            else:
                # Start next wave automatically in step function
                if self.wave_manager.start_next_wave():
                    self.wave_number = self.wave_manager.current_wave
                    self.player_money += 100  # Bonus money for completing a wave
                    reward += 200
                    self.current_wave_started = False
                    done = False  # Continue the episode
        
        done = self.game_over or (self.wave_number >= self.wave_manager.max_waves and self.wave_manager.wave_cleared and not self.enemies)
            
        # Display the game
        self.display()

        if DISPLAY_GAME:
            self.clock.tick(30)
        
        # Return step information
        return self.get_state(), reward, done, {}
    
    def get_performance_metrics(self):
        """Return comprehensive performance metrics for this episode"""
        # Calculate tower placement quality metrics
        strategic_positions = self.get_strategic_positions()
        close_positions = set(strategic_positions['close'])
        
        # Count towers by distance category
        close_towers = sum(1 for tower in self.towers if tower.grid_position in close_positions)
        
        # Calculate tower upgrade distribution
        tower_levels = [tower.level for tower in self.towers]
        avg_tower_level = sum(tower_levels) / max(1, len(tower_levels)) if tower_levels else 0
        upgraded_towers = sum(1 for level in tower_levels if level > 1)
        
        # Calculate overall path coverage
        path_cells = [(x, y) for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH) 
                    if self.game_map[y][x] in [1, 3, 5]]
        covered_path_cells = set()
        
        for tower in self.towers:
            for px, py in path_cells:
                tower_pos = tower.position
                path_pos = (px * GRID_SIZE + GRID_SIZE // 2, py * GRID_SIZE + GRID_SIZE // 2)
                dist = math.sqrt((tower_pos[0] - path_pos[0])**2 + (tower_pos[1] - path_pos[1])**2)
                if dist <= tower.range:
                    covered_path_cells.add((px, py))
        
        path_coverage = len(covered_path_cells) / max(1, len(path_cells))
        
        # Calculate economy efficiency
        money_efficiency = self.score / max(1, (self.player_money + sum(tower.level * TOWER_COST for tower in self.towers)))
        
        # Tower efficiency
        total_shots = sum(tower.shots_fired for tower in self.towers)
        tower_efficiency = total_shots / max(1, len(self.towers) * self.steps_taken / 30)
        
        return {
            'score': self.score,
            'waves_completed': self.wave_number,
            'lives_remaining': self.player_lives,
            'money_efficiency': money_efficiency,
            'tower_count': len(self.towers),
            'upgraded_tower_count': upgraded_towers,
            'avg_tower_level': avg_tower_level,
            'path_coverage': path_coverage,
            'tower_efficiency': tower_efficiency,
            'enemies_killed': self.enemies_killed,
            'enemies_leaked': self.enemies_leaked,
            'close_towers_ratio': close_towers / max(1, len(self.towers)) if self.towers else 0,
            'map_name': self.current_map,
            'steps_taken': self.steps_taken,
            'tower_positions': [(tower.grid_position, tower.level) for tower in self.towers],
            'total_reward': self.reward
        }