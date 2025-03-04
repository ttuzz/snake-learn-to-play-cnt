import pygame
from game import SnakeGame
from ai_model import SnakeAI
from direction import Direction
import argparse

def play_ai(model_path):
    game = SnakeGame()
    ai = SnakeAI()
    game.speed_multiplier=8

    # Modeli yükle
    try:
        ai.load(model_path)
        ai.epsilon = 0  # Keşif modunu kapat
        print(f"Model yüklendi: {model_path}")
    except:
        print(f"Model yüklenemedi: {model_path}")
        return

    # Oyun döngüsü
    while True:
        # Durumu al
        state = ai.get_state(game)
        
        # AI'nin hareketi
        action = ai.act(state)
        
        # Yönü güncelle
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(game.direction)
        
        if action == 0:  # Düz
            new_dir = clock_wise[idx]
        elif action == 1:  # Sağa dön
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # Sola dön
            new_dir = clock_wise[(idx - 1) % 4]
        
        game.direction = new_dir
        
        # Hareketi uygula
        game.move_snake()
        
        # Oyunu çiz
        game.draw()
        pygame.display.flip()
        
        # Oyun hızı
        game.clock.tick(game.speed)
        
        # Çıkış kontrolü
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
        
        # Oyun bitti mi?
        if game.game_over:
            print(f"Oyun bitti! Skor: {game.score}")
            game.reset_game()

import os
if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    else:
        checkpoint_files = [f for f in os.listdir("models") if f.endswith('.pth')]
        
        # Eğer checkpoint dosyası varsa, en son kaydedilen dosyayı yükle
        if checkpoint_files:
            # En son kaydedilen dosyayı seç (dosya adındaki çevrim numarasına göre)
            latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
            model_path = os.path.join("models", latest_checkpoint)

            # Çevrim numarasını çıkar (dosya adından)
            cycle_number = int(latest_checkpoint.split('_')[-1].split('.')[0])
            play_ai(f'models/model_checkpoint_{cycle_number}.pth') 
        else:
            print("Yeni bir model başlatılıyor.")
