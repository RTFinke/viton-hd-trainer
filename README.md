VITON-HD Training for CatVTON
=============================

Ten projekt pozwala na trenowanie modelu do wirtualnego przymierzania ubrań (VITON-HD) przy użyciu datasetu VITON-HD. Poniżej znajdziesz szczegółowe instrukcje, jak skonfigurować środowisko, przygotować dane i uruchomić trening, inference oraz ewaluację bez błędów.

Wymagania sprzętowe
------------------
- GPU: Karta NVIDIA z co najmniej 8GB VRAM (16GB+ zalecane). Dla GPU 8GB ustaw batch_size=1 i img_size=256.
- CPU: Wielordzeniowy procesor do ładowania danych.
- RAM: Minimum 16GB, zalecane 32GB+.
- Dysk: Minimum 20GB wolnego miejsca na dataset i checkpointy.

Wymagania oprogramowania
-----------------------
- Python 3.8
- CUDA 11.0+ (jeśli używasz GPU)
- Conda (do zarządzania środowiskiem)

Krok 1: Przygotowanie środowiska
--------------------------------

1. Zainstaluj Conda (jeśli nie masz):
   - Pobierz Miniconda z https://docs.conda.io/en/latest/miniconda.html i zainstaluj.
   - Sprawdź instalację:
     ```
     conda --version
     ```

2. Utwórz i aktywuj środowisko Conda:
   ```
   conda create -n vitonhd python=3.8 -y
   conda activate vitonhd
   ```

3. Zainstaluj zależności:
   - Upewnij się, że plik requirements.txt zawiera:
     ```
     numpy
     pillow
     tqdm
     wandb
     torch
     torchvision
     scikit-image
     ```
   - Zainstaluj zależności:
     ```
     pip install -r requirements.txt
     ```
   - Uwaga: Jeśli masz GPU, upewnij się, że wersja torch jest zgodna z CUDA. Sprawdź wersję CUDA:
     ```
     nvcc --version
     ```
     Dla CUDA 11.8 zainstaluj:
     ```
     pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
     ```

4. Zweryfikuj instalację:
   - Sprawdź import bibliotek:
     ```
     python -c "import torch, torchvision, numpy, PIL, tqdm, wandb, skimage; print('All dependencies imported successfully')"
     ```
   - Sprawdź dostępność GPU:
     ```
     python -c "import torch; print(torch.cuda.is_available())"
     ```
     Powinno zwrócić True dla GPU lub False dla CPU (trening na CPU będzie wolniejszy).

5. (Opcjonalne) Skonfiguruj WandB:
   - Jeśli chcesz logować wyniki do Weights & Biases:
     ```
     wandb login
     ```
   - Jeśli nie chcesz używać WandB, pomiń ten krok i nie dodawaj flagi --log_wandb podczas treningu.

Krok 2: Przygotowanie datasetu VITON-HD
--------------------------------------

1. Pobierz dataset VITON-HD:
   - Pobierz z https://github.com/sangyun884/VITON-HD lub innego zaufanego źródła (np. strony KAIST).
   - Dataset jest dostarczany jako plik .zip lub .tar.gz.

2. Rozpakuj dataset:
   - Rozpakuj archiwum do folderu ./data/ w głównym katalogu projektu:
     ```
     unzip VITON-HD.zip -d ./data
     ```
   - Oczekiwana struktura folderów:
     ```
     data/
     ├── train_img/        # person_001.jpg, person_002.jpg, ...
     ├── train_cloth/      # cloth_001.jpg, cloth_002.jpg, ...
     ├── train_cloth_mask/ # cloth_001.png, cloth_002.png, ...
     ├── train_parse/      # person_001.png, person_002.png, ...
     ├── test_img/         # person_test_001.jpg, ...
     ├── test_cloth/       # cloth_test_001.jpg, ...
     ├── test_cloth_mask/  # cloth_test_001.png, ...
     ├── test_parse/       # person_test_001.png, ...
     ```

3. Dostosuj strukturę (jeśli potrzeba):
   - Jeśli foldery mają inne nazwy (np. train/image/ zamiast train_img/), zmień nazwy:
     ```
     mv data/train/image data/train_img
     mv data/train/cloth data/train_cloth
     mv data/train/cloth_mask data/train_cloth_mask
     mv data/train/parse data/train_parse
     mv data/test/image data/test_img
     mv data/test/cloth data/test_cloth
     mv data/test/cloth_mask data/test_cloth_mask
     mv data/test/parse data/test_parse
     ```
   - Upewnij się, że:
     - Obrazy (train_img, train_cloth, test_img, test_cloth) są w formacie .jpg, .png lub .jpeg.
     - Maski i segmentacje (train_cloth_mask, train_parse, test_cloth_mask, test_parse) są w formacie .png.
     - Nazwy plików są spójne (np. person_001.jpg w train_img/ musi mieć odpowiadające cloth_001.jpg, cloth_001.png, person_001.png).

4. Zweryfikuj dataset:
   - Uruchom skrypt walidacyjny:
     ```
     python scripts/validate_data.py --dataroot ./data
     ```
   - Oczekiwany wynik:
     ```
     Train dataset: Found X valid samples
     Test dataset: Found Y valid samples
     Dataset validation passed!
     ```
   - Jeśli pojawią się błędy (np. "Missing parse file" lub "Directory not found"):
     - Sprawdź, czy wszystkie foldery istnieją.
     - Upewnij się, że nazwy plików są spójne.
     - Skonwertuj obrazy do właściwego formatu, jeśli to potrzebne (np. z .bmp na .jpg):
       ```
       mogrify -format jpg *.bmp
       ```

Krok 3: Struktura projektu
--------------------------

Upewnij się, że katalog projektu ma następującą strukturę:
```
viton-hd/
├── data/
│   ├── train_img/
│   ├── train_cloth/
│   ├── train_cloth_mask/
│   ├── train_parse/
│   ├── test_img/
│   ├── test_cloth/
│   ├── test_cloth_mask/
│   ├── test_parse/
├── models/
│   ├── __init__.py
│   ├── appearance_flow.py
│   ├── mask_generator.py
│   ├── networks.py
├── scripts/
│   ├── validate_data.py
├── checkpoints/       # Pusty, tutaj będą zapisywane checkpointy
├── results/          # Pusty, tutaj będą zapisywane wyniki inference
├── train_viton_hd.py
├── inference.py
├── evaluate.py
├── dataset.py
├── requirements.txt
├── README.txt
```

Krok 4: Trening modelu
----------------------

1. Uruchom trening:
   - Użyj polecenia, dostosowując batch_size i img_size do swojego GPU:
     ```
     python train_viton_hd.py \
       --dataroot ./data \
       --name vitonhd_run1 \
       --batch_size 2 \
       --img_size 256 \
       --epochs 50
     ```
   - Uwagi:
     - Dla GPU z 8GB VRAM ustaw batch_size=1 i img_size=256.
     - Jeśli chcesz logować wyniki do WandB, dodaj --log_wandb (po wandb login).
     - Checkpointy będą zapisywane w ./checkpoints/vitonhd_run1/ co 10 epok oraz na końcu treningu.

2. Monitoruj trening:
   - Skrypt wyświetla postępy w terminalu (paski postępu i metryki).
   - Sprawdź, czy dataset ładuje się poprawnie – powinieneś zobaczyć:
     ```
     Train dataset size: X, Test dataset size: Y
     ```
   - Jeśli używasz WandB, metryki (SSIM, PSNR, straty) będą widoczne w panelu WandB.

Krok 5: Inference
-----------------

1. Wygeneruj obrazy try-on:
   - Po wytrenowaniu modelu (np. po 50 epokach) uruchom:
     ```
     python inference.py \
       --checkpoint checkpoints/vitonhd_run1_epoch_49.pth \
       --dataroot ./data \
       --output_dir ./results \
       --img_size 256
     ```
   - Wyniki pojawią się w ./results/ jako pliki PNG w formacie <person_name>_<cloth_name>.png.

Krok 6: Ewaluacja
-----------------

1. Oceń model:
   - Uruchom ewaluację, aby obliczyć metryki SSIM i PSNR:
     ```
     python evaluate.py \
       --checkpoint checkpoints/vitonhd_run1_epoch_49.pth \
       --dataroot ./data \
       --batch_size 2 \
       --img_size 256
     ```
   - Skrypt wyświetli średnie wartości SSIM i PSNR.

Rozwiązywanie problemów
----------------------

1. Błąd: "No valid data samples found":
   - Uruchom:
     ```
     python scripts/validate_data.py --dataroot ./data
     ```
     Sprawdź brakujące pliki lub foldery.
   - Upewnij się, że nazwy plików są spójne i maski/segmentacje są w .png.

2. Błąd: "Out of memory":
   - Zmniejsz batch_size do 1 i/lub img_size do 256:
     ```
     python train_viton_hd.py --batch_size 1 --img_size 256 ...
     ```
   - Sprawdź użycie pamięci GPU:
     ```
     nvidia-smi
     ```

3. Błąd: Brak biblioteki:
   - Sprawdź zainstalowane zależności:
     ```
     pip list
     ```
   - Jeśli brakuje np. scikit-image, zainstaluj:
     ```
     pip install scikit-image
     ```

4. Błąd: WandB nie działa:
   - Wykonaj wandb login lub pomiń --log_wandb.

5. Błąd: Nieprawidłowy format pliku:
   - Skonwertuj obrazy do .jpg, jeśli mają inny format (np. .bmp):
     ```
     mogrify -format jpg *.bmp
     ```
   - Maski i segmentacje muszą być w .png.

6. Błąd: CUDA nie działa:
   - Sprawdź sterowniki NVIDIA i CUDA:
     ```
     nvidia-smi
     nvcc --version
     ```
   - Trenuj na CPU, jeśli CUDA nie działa:
     ```
     export CUDA_VISIBLE_DEVICES=""
     ```

Dodatkowe uwagi
---------------
- Czas treningu: Trening (50 epok, batch_size=2, img_size=256) może zająć kilka godzin, w zależności od GPU.
- Checkpointy: Sprawdzaj ./checkpoints/, czy modele są zapisywane.
- Wyniki: Po inference przejrzyj obrazy w ./results/, aby ocenić jakość.
- Dokumentacja: W razie wątpliwości zajrzyj do kodu w train_viton_hd.py, inference.py, evaluate.py.

Jeśli napotkasz problemy, zapisz szczegóły błędu i skontaktuj się z zespołem.