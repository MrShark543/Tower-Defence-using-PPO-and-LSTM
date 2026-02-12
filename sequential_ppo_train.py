import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pygame
import time
import math
import json
from collections import defaultdict, deque

import game
from game import GameManager, load_map


class StrategyTracker:
    """
    Tracks and analyzes strategic decision-making patterns of a PPO agent in a tower defense game.

    This class monitors various metrics over the course of training episodes, including tactical
    choices (e.g., tower placement and upgrades), map-specific behaviors, enemy responses, and
    wave-based strategies. It is used to extract interpretable insights and identify trends in 
    successful and failing strategies.

    Attributes:
        choke_point_usage (float): Ratio of towers placed near choke points.
        path_coverage (float): Proportion of path cells covered by tower ranges.
        tower_diversity (float): Diversity of tower levels used in the current episode.
        upgrade_vs_new_ratio (float): Ratio of upgraded towers vs newly placed towers.
        economy_management (float): Efficiency of money usage measured via reward-to-spending ratio.
        early_placements (list): First few tower placement decisions tracked across episodes.
        tower_placement_sequences (list): Sequences of tower placements per episode.
        response_to_enemy_types (dict): Records of actions taken in response to specific enemy types.
        map_performance (dict): Stores performance statistics per map.
        action_frequencies (dict): Counts of each action taken during training.
        wave_strategies (dict): Aggregated actions and stats by map and wave.
        successful_strategies (list): Snapshots of strategies that resulted in successful episodes.
        failed_strategies (list): Snapshots of strategies that led to failure.

    Methods:
        update_from_game_state(game_state, action, prev_state=None, reward=0):
            Updates internal metrics based on the current game state and the chosen action.
        
        record_episode_result(game_state, episode_reward, success):
            Stores high-level strategy and performance data at the end of an episode.
        
        get_strategic_insights():
            Analyzes collected data and returns interpretable insights on strategic trends,
            including enemy-specific responses, success/failure comparisons, and map difficulties.
    """
    def __init__(self):
        # General strategy metrics
        self.choke_point_usage = 0
        self.path_coverage = 0.0
        self.tower_diversity = 0.0
        self.upgrade_vs_new_ratio = 0.0
        self.economy_management = 0.0  # Measured by money saved vs spent
        
        # Decision tracking
        self.early_placements = []  # Store first N tower placements
        self.tower_placement_sequences = []  # Track order of tower placements
        self.response_to_enemy_types = {}  # Track responses to different enemy types
        
        # Map-specific metrics
        self.map_performance = {}  # Track performance by map
        
        # Temporal patterns
        self.action_frequencies = {}  # Track how often each action is taken
        self.wave_strategies = {}  # Track strategies per wave
        
        # Success patterns
        self.successful_strategies = []  # Store strategies that led to success
        self.failed_strategies = []  # Store strategies that led to failure
        
    def update_from_game_state(self, game_state, action, prev_state=None, reward=0):
        # Track action frequencies
        if action not in self.action_frequencies:
            self.action_frequencies[action] = 0
        self.action_frequencies[action] += 1
        
        # Skip further tracking if there are no towers
        if not game_state.towers:
            return
            
        # Track choke point usage
        strategic_positions = game_state.get_strategic_positions()
        close_positions = set(strategic_positions['close'])
        close_towers = sum(1 for tower in game_state.towers if tower.grid_position in close_positions)
        self.choke_point_usage = close_towers / len(game_state.towers)
        
        # Calculate path coverage
        path_cells = [(x, y) for y in range(game_state.GRID_HEIGHT) for x in range(game_state.GRID_WIDTH) 
                    if game_state.game_map[y][x] in [1, 3, 5]]
        covered_cells = 0
        for tower in game_state.towers:
            for px, py in path_cells:
                tower_pos = tower.grid_position
                path_pos = (px, py)
                dist = math.sqrt((tower_pos[0] - path_pos[0])**2 + 
                                (tower_pos[1] - path_pos[1])**2)
                if dist <= tower.range / game.GRID_SIZE:
                    covered_cells += 1
                    
        self.path_coverage = covered_cells / max(1, len(path_cells))
        
        # Track tower level diversity
        tower_levels = [tower.level for tower in game_state.towers]
        self.tower_diversity = len(set(tower_levels)) / max(1, len(tower_levels))
        
        # Track upgrade vs new tower ratio
        upgraded_towers = sum(1 for tower in game_state.towers if tower.level > 1)
        self.upgrade_vs_new_ratio = upgraded_towers / max(1, len(game_state.towers) - upgraded_towers)
        
        # Track economy management
        if prev_state:
            prev_money = prev_state['scalar_state'][0] * 500  # Unnormalize
            spent = prev_money - game_state.player_money
            if spent > 0:
                self.economy_management = reward / spent
        
        # Record early placement decisions (first 3 towers)
        if len(game_state.towers) <= 3 and action < 30:  # Tower placement actions
            self.early_placements.append((action, game_state.wave_number, game_state.current_map))
        
        # Track response to enemy types
        enemy_types_present = {enemy.enemy_type for enemy in game_state.enemies}
        for enemy_type in enemy_types_present:
            if enemy_type not in self.response_to_enemy_types:
                self.response_to_enemy_types[enemy_type] = []
            
            # Record if the agent placed or upgraded towers in response to this enemy type
            if action < 60:  # Tower placement or upgrade
                self.response_to_enemy_types[enemy_type].append(action)
        
        # Track wave-specific strategies
        wave_key = f"{game_state.current_map}_wave_{game_state.wave_number}"
        if wave_key not in self.wave_strategies:
            self.wave_strategies[wave_key] = {
                'actions': [],
                'towers_placed': 0,
                'towers_upgraded': 0,
                'avg_reward': 0,
                'count': 0
            }
        
        self.wave_strategies[wave_key]['actions'].append(action)
        self.wave_strategies[wave_key]['avg_reward'] = (
            (self.wave_strategies[wave_key]['avg_reward'] * self.wave_strategies[wave_key]['count'] + reward) /
            (self.wave_strategies[wave_key]['count'] + 1)
        )
        self.wave_strategies[wave_key]['count'] += 1
        
        # Track tower placements for this wave
        if action < 30:  # Tower placement
            self.wave_strategies[wave_key]['towers_placed'] += 1
        elif action < 60:  # Tower upgrade
            self.wave_strategies[wave_key]['towers_upgraded'] += 1
    
    def record_episode_result(self, game_state, episode_reward, success):
        """Record metrics at the end of an episode"""
        # Track map-specific performance
        map_name = game_state.current_map
        if map_name not in self.map_performance:
            self.map_performance[map_name] = {
                'attempts': 0,
                'successes': 0,
                'avg_reward': 0,
                'max_reward': float('-inf'),
                'avg_towers': 0,
                'avg_waves': 0,
                'avg_lives_remaining': 0
            }
        
        # Update map performance metrics
        stats = self.map_performance[map_name]
        stats['attempts'] += 1
        if success:
            stats['successes'] += 1
        
        # Update averages
        stats['avg_reward'] = ((stats['avg_reward'] * (stats['attempts'] - 1)) + 
                              episode_reward) / stats['attempts']
        stats['max_reward'] = max(stats['max_reward'], episode_reward)
        stats['avg_towers'] = ((stats['avg_towers'] * (stats['attempts'] - 1)) + 
                              len(game_state.towers)) / stats['attempts']
        stats['avg_waves'] = ((stats['avg_waves'] * (stats['attempts'] - 1)) + 
                             game_state.wave_number) / stats['attempts']
        stats['avg_lives_remaining'] = ((stats['avg_lives_remaining'] * 
                                      (stats['attempts'] - 1)) + 
                                     game_state.player_lives) / stats['attempts']
        
        # Record successful strategies
        strategy_snapshot = {
            'map': game_state.current_map,
            'reward': episode_reward,
            'towers': [(t.grid_position, t.level, t.shots_fired) for t in game_state.towers],
            'tower_count': len(game_state.towers),
            'choke_point_usage': self.choke_point_usage,
            'path_coverage': self.path_coverage,
            'tower_diversity': self.tower_diversity,
            'upgrade_ratio': self.upgrade_vs_new_ratio,
            'lives_remaining': game_state.player_lives,
            'wave_reached': game_state.wave_number
        }
        
        if success:
            self.successful_strategies.append(strategy_snapshot)
        else:
            self.failed_strategies.append(strategy_snapshot)
    
    def get_strategic_insights(self):
        """Extract key strategic insights from collected data"""
        insights = {}
        
        # Analyze common patterns in successful strategies
        if self.successful_strategies:
            successful_maps = {}
            for strat in self.successful_strategies:
                map_name = strat['map']
                if map_name not in successful_maps:
                    successful_maps[map_name] = []
                successful_maps[map_name].append(strat)
            
            insights['successful_maps'] = {
                map_name: len(strategies) for map_name, strategies in successful_maps.items()
            }
            
            # Analyze tower placement patterns in successful runs
            if len(self.successful_strategies) >= 3:
                avg_tower_count = sum(s['tower_count'] for s in self.successful_strategies) / len(self.successful_strategies)
                avg_choke_usage = sum(s['choke_point_usage'] for s in self.successful_strategies) / len(self.successful_strategies)
                avg_path_coverage = sum(s['path_coverage'] for s in self.successful_strategies) / len(self.successful_strategies)
                avg_tower_diversity = sum(s['tower_diversity'] for s in self.successful_strategies) / len(self.successful_strategies)
                
                insights['successful_strategies'] = {
                    'avg_tower_count': avg_tower_count,
                    'avg_choke_point_usage': avg_choke_usage,
                    'avg_path_coverage': avg_path_coverage,
                    'avg_tower_diversity': avg_tower_diversity
                }
        
        # Compare with failing strategies
        if self.successful_strategies and self.failed_strategies:
            succ_towers = sum(s['tower_count'] for s in self.successful_strategies) / max(1, len(self.successful_strategies))
            fail_towers = sum(s['tower_count'] for s in self.failed_strategies) / max(1, len(self.failed_strategies))
            
            succ_choke = sum(s['choke_point_usage'] for s in self.successful_strategies) / max(1, len(self.successful_strategies))
            fail_choke = sum(s['choke_point_usage'] for s in self.failed_strategies) / max(1, len(self.failed_strategies))
            
            insights['strategy_comparison'] = {
                'tower_count_diff': succ_towers - fail_towers,
                'choke_usage_diff': succ_choke - fail_choke,
                'upgrade_tendency': {
                    'success': sum(s['upgrade_ratio'] for s in self.successful_strategies) / max(1, len(self.successful_strategies)),
                    'failure': sum(s['upgrade_ratio'] for s in self.failed_strategies) / max(1, len(self.failed_strategies))
                }
            }
            
        # Analyze enemy-specific responses
        if self.response_to_enemy_types:
            insights['enemy_responses'] = {}
            for enemy_type, actions in self.response_to_enemy_types.items():
                if actions:
                    # Count actions by type (placement vs upgrade)
                    placements = sum(1 for a in actions if a < 30)
                    upgrades = sum(1 for a in actions if 30 <= a < 60)
                    wait = sum(1 for a in actions if a >= 60)
                    
                    insights['enemy_responses'][enemy_type] = {
                        'placement_rate': placements / len(actions),
                        'upgrade_rate': upgrades / len(actions),
                        'wait_rate': wait / len(actions),
                        'total_actions': len(actions)
                    }
        
        return insights

# Track curriculum progression function
def track_curriculum_progression(episode, current_map, successful_completions, map_order):
    """Track progression through curriculum learning"""
    progression_metrics = {
        'episode': episode,
        'current_map': current_map,
        'map_index': map_order.index(current_map),
        'successful_completions': successful_completions,
        'normalized_progression': map_order.index(current_map) / max(1, len(map_order) - 1),
        'timestamp': time.time()
    }
    return progression_metrics

# Create comprehensive training report function
def create_comprehensive_training_report(agent, training_data, curriculum_progression, strategy_tracker):
    """Create comprehensive visualizations for the report"""
    # Create directory for report plots
    report_dir = "training_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Overall performance over time
    plt.figure(figsize=(15, 15))
    
    plt.subplot(3, 2, 1)
    plt.plot(training_data['episode_rewards'], label='Episode Reward')
    plt.title('Training Reward Progression')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    

    map_changes = []
    for i in range(1, len(curriculum_progression)):
        if curriculum_progression[i]['map_index'] != curriculum_progression[i-1]['map_index']:
            map_changes.append(i)
            plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)
            plt.text(i+1, min(training_data['episode_rewards']) + 50, 
                    f"Map {curriculum_progression[i]['map_index']+1}", 
                    rotation=90, verticalalignment='bottom')
    
    # Plot moving average
    if len(training_data['episode_rewards']) > 10:
        window_size = min(25, len(training_data['episode_rewards']) // 10)
        moving_avg = np.convolve(training_data['episode_rewards'], 
                               np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, window_size-1+len(moving_avg)), 
                moving_avg, 'r-', linewidth=2, label=f'{window_size}-ep Moving Avg')
    plt.legend()
    
    #  Wave completion progress
    plt.subplot(3, 2, 2)
    plt.plot(training_data['wave_progress'], 'g-', label='Max Wave')
    plt.title('Wave Completion Progress')
    plt.xlabel('Episode')
    plt.ylabel('Max Wave Reached')
    plt.grid(True)
    
    for i in map_changes:
        plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)
    
    # Success rate progression
    plt.subplot(3, 2, 3)
    # Calculate moving success rate
    window = 20
    if len(training_data['success_history']) >= window:
        success_rate = []
        for i in range(len(training_data['success_history']) - window + 1):
            rate = sum(training_data['success_history'][i:i+window]) / window
            success_rate.append(rate)
        plt.plot(range(window-1, len(training_data['success_history'])), 
                success_rate, 'b-', linewidth=2)
        plt.title(f'Success Rate ({window}-episode window)')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1.05)
        plt.grid(True)
    
    # Map progression over time
    plt.subplot(3, 2, 4)
    map_indices = [item['map_index'] for item in curriculum_progression]
    plt.plot(map_indices, 'b-', linewidth=2)
    plt.yticks(range(len(training_data['map_order'])), training_data['map_order'])
    plt.title('Curriculum Progression')
    plt.xlabel('Episode')
    plt.ylabel('Map')
    plt.grid(True)
    
    #  Architecture contribution
    if 'architecture_contributions' in training_data and training_data['architecture_contributions']:
        plt.subplot(3, 2, 5)
        episodes = range(len(training_data['architecture_contributions']))
        lstm_contrib = [c['lstm_contribution'] for c in training_data['architecture_contributions']]
        cnn_contrib = [c['cnn_contribution'] for c in training_data['architecture_contributions']]
        scalar_contrib = [c['scalar_contribution'] for c in training_data['architecture_contributions']]
        
        plt.stackplot(episodes, lstm_contrib, cnn_contrib, scalar_contrib,
                    labels=['LSTM', 'CNN', 'Scalar Features'],
                    colors=['#ff9999','#66b3ff','#99ff99'])
        plt.title('Neural Network Component Contributions')
        plt.xlabel('Policy Update')
        plt.ylabel('Contribution Ratio')
        plt.legend(loc='upper left')
        plt.grid(True)
    
    # Exploration vs exploitation
    if 'exploration_metrics' in training_data and training_data['exploration_metrics']:
        plt.subplot(3, 2, 6)
        episodes = range(len(training_data['exploration_metrics']))
        action_entropy = [m['action_entropy'] for m in training_data['exploration_metrics']]
        plt.plot(episodes, action_entropy, 'purple', label='Action Diversity')
        
        if len(action_entropy) > 10:
            window_size = min(10, len(action_entropy) // 5)
            smoothed = np.convolve(action_entropy, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed)), 
                    smoothed, 'r-', linewidth=2, label=f'{window_size}-pt Moving Avg')
        
        plt.title('Exploration vs Exploitation')
        plt.xlabel('Episode')
        plt.ylabel('Action Entropy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{report_dir}/overall_performance_{timestamp}.png", dpi=150)
    plt.close()
    
    if strategy_tracker and hasattr(strategy_tracker, 'map_performance'):
        plt.figure(figsize=(15, 10))
        
        # Plot map-specific performance
        plt.subplot(2, 2, 1)
        maps = list(strategy_tracker.map_performance.keys())
        success_rates = [strategy_tracker.map_performance[m]['successes'] / 
                        max(1, strategy_tracker.map_performance[m]['attempts']) 
                        for m in maps]
        
        plt.bar(maps, success_rates, color='green')
        plt.title('Success Rate by Map')
        plt.xlabel('Map')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1.05)
        plt.grid(axis='y')
        
        # Plot tower placement strategies
        plt.subplot(2, 2, 2)
        if hasattr(strategy_tracker, 'action_frequencies') and strategy_tracker.action_frequencies:
            # Group actions by type
            place_close = sum(strategy_tracker.action_frequencies.get(i, 0) for i in range(0, 10))
            place_mid = sum(strategy_tracker.action_frequencies.get(i, 0) for i in range(10, 20))
            place_far = sum(strategy_tracker.action_frequencies.get(i, 0) for i in range(20, 30))
            upgrade_close = sum(strategy_tracker.action_frequencies.get(i, 0) for i in range(30, 40))
            upgrade_mid = sum(strategy_tracker.action_frequencies.get(i, 0) for i in range(40, 50))
            upgrade_far = sum(strategy_tracker.action_frequencies.get(i, 0) for i in range(50, 60))
            wait = sum(strategy_tracker.action_frequencies.get(i, 0) for i in range(60, 61))
            
            labels = ['Place\nClose', 'Place\nMid', 'Place\nFar', 
                     'Upgrade\nClose', 'Upgrade\nMid', 'Upgrade\nFar', 'Wait']
            values = [place_close, place_mid, place_far, 
                     upgrade_close, upgrade_mid, upgrade_far, wait]
            
            total = sum(values)
            if total > 0:
                normalized = [v/total for v in values]
                plt.bar(labels, normalized)
                plt.title('Action Distribution')
                plt.ylabel('Frequency')
                plt.grid(axis='y')
        
        # Plot strategic insights
        if hasattr(strategy_tracker, 'get_strategic_insights'):
            insights = strategy_tracker.get_strategic_insights()
            
            if 'successful_strategies' in insights:
                plt.subplot(2, 2, 3)
                metrics = insights['successful_strategies']
                labels = ['Tower Count', 'Choke Usage', 'Path Coverage', 'Tower Diversity']
                values = [metrics['avg_tower_count']/10, 
                         metrics['avg_choke_point_usage'], 
                         metrics['avg_path_coverage'],
                         metrics['avg_tower_diversity']]
                
                plt.bar(labels, values, color='blue')
                plt.title('Successful Strategy Metrics')
                plt.ylabel('Average Value')
                plt.grid(axis='y')
            
            # Enemy response patterns
            if 'enemy_responses' in insights and insights['enemy_responses']:
                plt.subplot(2, 2, 4)
                enemy_types = list(insights['enemy_responses'].keys())
                placement_rates = [insights['enemy_responses'][e]['placement_rate'] for e in enemy_types]
                upgrade_rates = [insights['enemy_responses'][e]['upgrade_rate'] for e in enemy_types]
                wait_rates = [insights['enemy_responses'][e]['wait_rate'] for e in enemy_types]
                
                x = np.arange(len(enemy_types))
                width = 0.25
                
                plt.bar(x - width, placement_rates, width, label='Place')
                plt.bar(x, upgrade_rates, width, label='Upgrade')
                plt.bar(x + width, wait_rates, width, label='Wait')
                
                plt.xlabel('Enemy Type')
                plt.ylabel('Action Rate')
                plt.title('Response to Enemy Types')
                plt.xticks(x, enemy_types)
                plt.legend()
                plt.grid(axis='y')
        
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(f"{report_dir}/strategy_analysis_{timestamp}.png", dpi=150)
        plt.close()
    
    # Create convergence analysis
    if 'convergence_metrics' in training_data and training_data['convergence_metrics']:
        plt.figure(figsize=(15, 5))
        
        # Plot convergence metrics
        plt.subplot(1, 2, 1)
        episodes = range(len(training_data['convergence_metrics']))
        return_var = [m['return_variance'] for m in training_data['convergence_metrics']]
        plt.plot(episodes, return_var, 'r-', label='Return Variance')
        
        if len(return_var) > 10:
            window_size = min(10, len(return_var) // 5)
            smoothed = np.convolve(return_var, 
                                np.ones(window_size)/window_size, 
                                mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(smoothed)), 
                    smoothed, 'b-', linewidth=2, 
                    label=f'{window_size}-pt Moving Avg')
        
        plt.title('Training Stability: Return Variance')
        plt.xlabel('Episode')
        plt.ylabel('Variance')
        plt.legend()
        plt.grid(True)
        
        # Plot policy and value losses
        plt.subplot(1, 2, 2)
        policy_losses = [np.mean(training_data['policy_losses'][max(0, i-10):i+1]) 
                       for i in range(len(training_data['policy_losses']))]
        value_losses = [np.mean(training_data['value_losses'][max(0, i-10):i+1]) 
                      for i in range(len(training_data['value_losses']))]
        
        plt.plot(policy_losses, 'r-', label='Policy Loss')
        plt.plot(value_losses, 'b-', label='Value Loss')
        plt.title('Training Losses')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{report_dir}/convergence_analysis_{timestamp}.png", dpi=150)
    
    # Save raw data for future analysis
    with open(f"{report_dir}/training_data_{timestamp}.json", 'w') as f:
        json_data = {
            'episode_rewards': training_data['episode_rewards'],
            'wave_progress': training_data['wave_progress'],
            'success_rate': training_data.get('success_history', []),
            'map_progression': [m['map_index'] for m in curriculum_progression],
            'successful_completions': [m['successful_completions'] for m in curriculum_progression],
            'map_names': training_data['map_order']
        }
        json.dump(json_data, f)
    
    return f"{report_dir}/overall_performance_{timestamp}.png"

# Analyze struggle scenarios function
def analyze_struggle_scenarios( performance_metrics):
    """Identify and analyze scenarios where the agent consistently struggled"""
    mean_reward = np.mean([m.get('total_reward', 0) for m in performance_metrics])
    poor_threshold = mean_reward * 0.5  # 50% below average is considered poor
    poor_episodes = [i for i, m in enumerate(performance_metrics) 
                   if m.get('total_reward', 0) < poor_threshold]
    
    # Group by map type
    map_struggle_count = defaultdict(int)
    map_completion_rates = defaultdict(list)
    
    for ep_idx in poor_episodes:
        map_name = performance_metrics[ep_idx]['map_name']
        map_struggle_count[map_name] += 1
    
    # Calculate map completion rates
    for metrics in performance_metrics:
        map_name = metrics.get('map_name')
        if map_name:
            success = metrics.get('waves_completed', 0) >= 3 and metrics.get('lives_remaining', 0) > 0
            map_completion_rates[map_name].append(1 if success else 0)
    
    map_success_rates = {
        map_name: sum(rates)/max(1, len(rates)) 
        for map_name, rates in map_completion_rates.items()
    }
    
    # Analyze tower configurations in failures
    tower_configs = defaultdict(list)
    for i in poor_episodes:
        if i < len(performance_metrics):
            metrics = performance_metrics[i]
            tower_count = metrics.get('tower_count', 0)
            upgraded = metrics.get('upgraded_tower_count', 0)
            config = f"{tower_count}T-{upgraded}U"
            tower_configs[config].append(i)
    
    # Analyze wave progression patterns
    wave_progression = defaultdict(list)
    for i in poor_episodes:
        if i < len(performance_metrics):
            map_name = performance_metrics[i]['map_name']
            max_wave = performance_metrics[i].get('waves_completed', 0)
            wave_progression[map_name].append(max_wave)
    
    avg_wave_progression = {
        map_name: sum(waves)/max(1, len(waves))
        for map_name, waves in wave_progression.items()
    }
    
    # Analyze common patterns in tower placement
    placement_patterns = {}
    for i in poor_episodes:
        if i < len(performance_metrics) and 'tower_positions' in performance_metrics[i]:
            map_name = performance_metrics[i]['map_name']
            if map_name not in placement_patterns:
                placement_patterns[map_name] = {
                    'close_ratio': 0,
                    'medium_ratio': 0,
                    'far_ratio': 0,
                    'count': 0
                }
            
            if 'close_towers_ratio' in performance_metrics[i]:
                placement_patterns[map_name]['close_ratio'] += performance_metrics[i]['close_towers_ratio']
            
            placement_patterns[map_name]['count'] += 1
    
    # Calculate average placement ratios
    for map_name, data in placement_patterns.items():
        if data['count'] > 0:
            data['close_ratio'] /= data['count']
    
    # Identify the most problematic maps
    sorted_struggles = sorted(map_struggle_count.items(), key=lambda x: x[1], reverse=True)
    top_struggle_maps = sorted_struggles[:3] if len(sorted_struggles) >= 3 else sorted_struggles
    
    # Identify the most common inadequate tower configurations
    sorted_configs = sorted(tower_configs.items(), key=lambda x: len(x[1]), reverse=True)
    top_struggle_configs = sorted_configs[:3] if len(sorted_configs) >= 3 else sorted_configs
    
    return {
        'poor_episodes': poor_episodes,
        'map_struggle_count': dict(map_struggle_count),
        'map_success_rates': map_success_rates,
        'top_struggle_maps': top_struggle_maps,
        'tower_configuration_struggles': {config: len(episodes) for config, episodes in dict(top_struggle_configs).items()},
        'avg_wave_progression': avg_wave_progression,
        'placement_patterns': placement_patterns
    }

# Main training function
def train_sequential(num_episodes=2000, render_interval=100, required_successes=3):
    """
    Train the PPO agent sequentially on maps, advancing only after achieving
    the required number of successful completions on the current map.
    
    Args:
        num_episodes: Maximum number of training episodes
        render_interval: How often to render the game
        required_successes: Number of successful completions needed to advance
    """
    # Turn off game display for training
    game.DISPLAY_GAME = False
    recent_maps_buffer = deque(maxlen=3)
    # Create game environment
    env = GameManager()
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create agent
    from ppo_lstm_agent import PPOLSTMAgent
    action_size = 3 * 10 + 3 * 10 + 1  # Places, upgrades, and do nothing
    try:
        agent = PPOLSTMAgent(action_size, device, scalar_size=6)  # Specify scalar_size=6
        print("Successfully created PPOLSTMAgent")
    except Exception as e:
        print(f"Error creating PPOLSTMAgent: {e}")
        return None
    
    # Map progression (ordered by presumed difficulty)
    map_order = ["map1_0", "map1_1", "map1_2", "map1_3", "map1_4", "map1_5", "map1_6", "map2_0", "map2_1", "map2_2"]
    current_map_index = 0
    
    # Tracking map completions
    successful_completions = 0
    
    # Initialize strategy tracker
    strategy_tracker = StrategyTracker()
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    wave_progress = []
    lives_remaining = []
    map_history = []
    success_count_history = []
    curriculum_progression = []
    performance_metrics = []
    success_history = []
    all_actions = []
    all_rewards = []
    
    # Architecture contribution analysis
    architecture_contributions = []
    convergence_metrics = []
    exploration_metrics = []
    
    # Loss tracking
    policy_losses = []
    value_losses = []
    entropy_losses = []
    
    # Create directories for models and plots
    os.makedirs("models", exist_ok=True)
    os.makedirs("training_plots", exist_ok=True)
    
    best_reward = -float('inf')
    
    # Training loop
    for episode in range(1, num_episodes + 1):
        # Add map to recent buffer
        recent_maps_buffer.append(env.current_map)
        # Set current map
        current_map = map_order[current_map_index]
        map_history.append(current_map)
        success_count_history.append(successful_completions)
        
        # Track curriculum progression
        curriculum_progression.append(track_curriculum_progression(
            episode, current_map, successful_completions, map_order))
        
        env.current_map = current_map
        env.game_map, env.all_paths, env.paths = load_map(env.current_map)
        
        # Set maximum waves for this map (can be adjusted per map if needed)
        env.max_waves = 3  
        
        # Reset environment and agent
        env.reset_game()
        agent.reset_hidden_states()
        
        # Get initial state
        state = env.get_state()
        prev_state = None
        
        # Episode tracking
        episode_reward = 0
        episode_experiences = []
        steps = 0
        done = False
        max_wave = 1
        episode_actions = []

        # Set render flag
        game.DISPLAY_GAME = (episode % render_interval == 0)

        if game.DISPLAY_GAME:
            print(f"Episode {episode} - RENDERING ENABLED - Map: {current_map}")

        # Episode loop
        while not done:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Track highest wave
            max_wave = max(max_wave, env.wave_number)
            
            # Select action
            action, action_prob, value = agent.select_action(state)
            
            # Record the action
            episode_actions.append(action)
            all_actions.append(action)
            
            # Track strategy before taking action
            strategy_tracker.update_from_game_state(env, action, prev_state, reward=0)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Record reward
            all_rewards.append(reward)
            
            # Store experience as dictionary
            experience = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'action_prob': action_prob,
                'value': value
            }
            
            episode_experiences.append(experience)
            
            # Update state and reward
            prev_state = state
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Check if the game is over
        episode_successful = not env.game_over and env.wave_number >= env.max_waves
        
        # Record success/failure
        success_history.append(1 if episode_successful else 0)
        
        # Record episode performance metrics
        perf_metrics = env.get_performance_metrics()
        performance_metrics.append(perf_metrics)
        
        # Update strategy tracker with episode result
        strategy_tracker.record_episode_result(env, episode_reward, episode_successful)
        
        if episode_successful:
            successful_completions += 1
            print(f"Successfully completed all waves on {current_map}! " +
                  f"({successful_completions}/{required_successes})")
        else:
            print(f" Failed to complete all waves on {current_map}. " +
                  f"Reached wave {env.wave_number} with {env.player_lives} lives left.")
        
  

        if successful_completions >= required_successes:
    # Cycle back to the first map if we're at the end
            if current_map_index >= len(map_order) - 1:
                print("\n COMPLETED ALL MAPS! Cycling back to first map to continue mastery.\n")
                current_map_index = 0
            else:
                current_map_index += 1
            
            successful_completions = 0  # Reset counter for new map
        
        # Add all experiences to buffer
        for exp in episode_experiences:
            agent.trajectory_buffer.push(exp)
        
        # Update policy if enough data
        if len(agent.trajectory_buffer) >= agent.batch_size:
            update_info = agent.update_policy(episode_experiences)
            print(f"Policy update - Loss: {update_info['total_loss']:.4f}, Return: {update_info['avg_return']:.2f}")
            
            # Track architecture contributions
            arch_contribution = agent.analyze_architectural_contribution()
            architecture_contributions.append(arch_contribution)
            
            # Track convergence metrics
            conv_metrics = agent.calculate_convergence_metrics()
            convergence_metrics.append(conv_metrics)
            
            # Track exploration/exploitation metrics
            expl_metrics = agent.track_exploration_exploitation(all_actions, all_rewards)
            exploration_metrics.append(expl_metrics)
            
            # Track losses
            policy_losses.append(update_info['policy_loss'])
            value_losses.append(update_info['value_loss'])
            entropy_losses.append(update_info['entropy'])
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        wave_progress.append(max_wave)
        lives_remaining.append(env.player_lives)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(f"models/ppo_sequential_best_model.pth")
        
        # Save checkpoint
        if episode % 100 == 0:
            agent.save(f"models/ppo_sequential_model_episode_{episode}.pth")
            
            # Create intermediate training report
            training_data = {
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths,
                'wave_progress': wave_progress,
                'lives_remaining': lives_remaining,
                'map_order': map_order,
                'architecture_contributions': architecture_contributions,
                'convergence_metrics': convergence_metrics,
                'exploration_metrics': exploration_metrics,
                'success_history': success_history,
                'policy_losses': policy_losses,
                'value_losses': value_losses,
                'entropy_losses': entropy_losses
            }
            
            # Generate comprehensive report
            report_path = create_comprehensive_training_report(
                agent, training_data, curriculum_progression, strategy_tracker)
            
            # Analyze struggle scenarios
            struggles = analyze_struggle_scenarios( performance_metrics)
            
            # Log struggle analysis
            print(f"\nStruggle Analysis at Episode {episode}:")
            print(f"Top struggle maps: {struggles['top_struggle_maps']}")
            print(f"Map success rates: {struggles['map_success_rates']}")
            print(f"Problematic tower configurations: {struggles['tower_configuration_struggles']}")
            print(f"Average wave progression: {struggles['avg_wave_progression']}")
        
        # Print progress
        print(f"Episode {episode}/{num_episodes} - Map: {current_map}, " +
              f"Reward: {episode_reward:.2f}, Steps: {steps}, Lives: {env.player_lives}, " +
              f"Max Wave: {max_wave}")
        
        # Plot progress every 10 episodes
        if episode % 10 == 0:
            plot_sequential_progress(
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                wave_progress=wave_progress,
                lives_remaining=lives_remaining,
                map_history=map_history,
                success_count_history=success_count_history,
                curr_episode=episode,
                map_order=map_order
            )
    
    # Save final model
    agent.save("models/ppo_sequential_final_model.pth")
    
    # Create final comprehensive report
    final_training_data = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'wave_progress': wave_progress,
        'lives_remaining': lives_remaining,
        'map_order': map_order,
        'architecture_contributions': architecture_contributions,
        'convergence_metrics': convergence_metrics,
        'exploration_metrics': exploration_metrics,
        'success_history': success_history,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'entropy_losses': entropy_losses
    }
    
    final_report_path = create_comprehensive_training_report(
        agent, final_training_data, curriculum_progression, strategy_tracker)
    
    print(f"Final comprehensive training report created: {final_report_path}")
    
    return agent

def plot_sequential_progress(episode_rewards, episode_lengths, wave_progress, 
                           lives_remaining, map_history, success_count_history,
                           curr_episode, map_order):
    """Plot training progress with map progression information"""
    # Create directory for plots
    plots_dir = "training_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Plot training metrics
    plt.figure(figsize=(15, 12))
    
    # Plot rewards
    plt.subplot(3, 2, 1)
    plt.plot(episode_rewards)
    
    # Add vertical lines for map changes
    map_changes = []
    for i in range(1, len(map_history)):
        if map_history[i] != map_history[i-1]:
            map_changes.append(i)
            plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Plot waves completed
    plt.subplot(3, 2, 2)
    plt.plot(wave_progress, 'g-')
    
    # Add map change lines
    for i in map_changes:
        plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Waves Completed per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Wave Number')
    plt.grid(True)
    
    # Plot episode lengths
    plt.subplot(3, 2, 3)
    plt.plot(episode_lengths, 'r-')
    plt.title('Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    # Plot lives remaining
    plt.subplot(3, 2, 4)
    plt.plot(lives_remaining, 'orange')
    plt.title('Lives Remaining at Episode End')
    plt.xlabel('Episode')
    plt.ylabel('Lives')
    plt.grid(True)
    
    # Plot map progression
    plt.subplot(3, 2, 5)
    
    # Convert map names to numeric indices for plotting
    map_indices = [map_order.index(m) for m in map_history]
    plt.plot(map_indices, 'b-')
    plt.yticks(range(len(map_order)), map_order)
    plt.title('Map Progression')
    plt.xlabel('Episode')
    plt.ylabel('Current Map')
    plt.grid(True)
    
    # Plot success counter
    plt.subplot(3, 2, 6)
    plt.plot(success_count_history, 'purple')
    plt.title('Successful Completions Counter')
    plt.xlabel('Episode')
    plt.ylabel('Success Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/ppo_sequential_progress_episode_{curr_episode}.png", dpi=150)
    plt.close()

