# ViViT Video Classifier – Instrukcja uruchomienia (kamera lokalna)

System do analizy wideo w czasie rzeczywistym z wykorzystaniem modelu ViViT. Źródłem obrazu jest kamera urządzenia, na którym otwarta jest strona (telefon/komputer). Bez Vdo.Ninja.

## 🎯 Architektura

```
Telefon (Android) → Kamera → Przeglądarka → Komputer z CUDA (ViViT) → Wynik → Wyświetlanie
                                    ↓
                          Opcjonalnie: VPS z Nginx (reverse proxy)
```

## 📋 Wymagania

### Komputer z CUDA:
- Python 3.8+
- NVIDIA GPU z CUDA
- 8GB+ RAM
- Zainstalowany CUDA Toolkit i cuDNN

### VPS (opcjonalnie):
- Nginx
- Dostęp SSH

### Telefon/Urządzenie:
- Przeglądarka z obsługą kamery (Chrome/Firefox)

## 🚀 Instalacja i uruchomienie

### 1️⃣ Na komputerze z CUDA:

```bash
pip install -r requirements.txt
python main.py
```

Przy pierwszym uruchomieniu model ViViT (~330MB) zostanie automatycznie pobrany.

### 2️⃣ Konfiguracja firewall (na komputerze z CUDA):

Otwórz port 5000:

**Windows:**
```powershell
netsh advfirewall firewall add rule name="ViViT Server" dir=in action=allow protocol=TCP localport=5000
```

**Linux:**
```bash
sudo ufw allow 5000/tcp
```

### 3️⃣ Dostęp z telefonu

**Opcja A: WireGuard (rekomendowane)**
- Włącz WireGuard na telefonie i komputerze
- Otwórz: `http://IP_WIREGUARD_KOMPUTERA:5000`

**Opcja B: Ta sama sieć WiFi**
- Otwórz: `http://IP_LOKALNE_KOMPUTERA:5000`

**Opcja C: Przez VPS z Nginx**
- Otwórz: `http://twoja-domena.com` lub `http://IP_VPS`
- VPS reverse proxy do komputera z CUDA

## 🎮 Użytkowanie

1. Otwórz stronę na telefonie/komputerze
2. Zezwól na dostęp do kamery
3. Kliknij „Rozpocznij analizę”
4. System wyświetli:
   - Nazwę wykrytej klasy
   - Poziom pewności (0–100%)

## 📊 Informacje techniczne

- Model: Google ViViT-B-16x2 (Kinetics-400)
- Klasy: 400 aktywności
- Częstotliwość: ~2 predykcje/sekundę
- Bufor: 32 ramki
- Rozdzielczość wejścia: 224×224 px

## 🔧 Rozwiązywanie problemów

### Brak GPU/CUDA
System automatycznie użyje CPU (wolniej).

### „Brak połączenia z serwerem”
- Sprawdź `http://localhost:5000/health`
- Firewall i port 5000

### Wolne działanie
- Upewnij się, że używasz GPU („Urządzenie: cuda” w logach)
- W `templates/index.html` zwiększ interwał wysyłania (domyślnie 100 ms)

### Błędy kamery
**Komunikat:** `cannot read properties of undefined (reading 'getUserMedia')`

Przeglądarka blokuje dostęp do kamery, jeśli strona nie jest w bezpiecznym kontekście.

- Wymagany jest HTTPS lub `localhost` (secure context)
- Rozwiązania:
    - Uruchom stronę na tym samym komputerze i otwórz `http://localhost:5000`
    - Skonfiguruj HTTPS na VPS (Nginx + Certbot) i otwieraj `https://twoja-domena.com`
    - Alternatywnie użyj Cloudflare Tunnel, aby uzyskać publiczny HTTPS bez otwierania portów
  
- Sprawdź uprawnienia przeglądarki do kamery

## 🔒 Bezpieczeństwo

**WireGuard:**
- Ruch szyfrowany przez tunel VPN; HTTPS nie wymagane

**HTTPS lokalnie na Windows (bez VPS):**
Możesz uruchomić serwer Flask bezpośrednio z HTTPS, korzystając z lokalnie zaufanego certyfikatu.

1) Zainstaluj mkcert (Windows, PowerShell):
```powershell
choco install mkcert -y
mkcert -install
```

2) Wygeneruj certyfikat dla hosta/IP (wybierz właściwy):
```powershell
# dla localhost
mkcert localhost

# dla IP WireGuard (przykład)
mkcert 10.0.0.2

# dla lokalnego IP w LAN (przykład)
mkcert 192.168.1.20
```
Powstaną pliki, np.: `localhost.pem` i `localhost-key.pem`.

3) Uruchom serwer Flask z HTTPS:
```powershell
$env:ENABLE_HTTPS = "true"
$env:SSL_CERT_PATH = "c:\\Users\\damia\\Desktop\\ViViT Server\\localhost.pem"
$env:SSL_KEY_PATH  = "c:\\Users\\damia\\Desktop\\ViViT Server\\localhost-key.pem"
python main.py
```

4) Otwórz stronę:
- Na tym komputerze: `https://localhost:5000`
- Z telefonu: `https://IP_WIREGUARD:5000` lub `https://IP_LAN:5000`

Uwaga (Android): Aby certyfikat był zaufany na telefonie, zainstaluj główny certyfikat CA mkcert na urządzeniu (Ustawienia → Bezpieczeństwo → Zainstaluj certyfikat → CA). Alternatywnie użyj `localhost` na tym samym urządzeniu lub WireGuard.

## 📞 Wydajność

**GPU (RTX 3060+):**
- Ładowanie modelu: ~5–10 s
- Inference: ~0.5 s/predykcja

**CPU:**
- Ładowanie modelu: ~10–20 s
- Inference: ~3–5 s/predykcja

## 🛠️ Dostosowanie

### Zmiana modelu
W [main.py](main.py):
```python
MODEL_NAME = "google/vivit-b-16x2-kinetics400"
```

### Rozmiar bufora
```python
FRAME_BUFFER_SIZE = 32
```

### Częstotliwość predykcji
W `run_inference()`:
```python
time.sleep(0.5)
```

## 📦 Struktura projektu

```
ViViT Server/
├── main.py
├── requirements.txt
├── nginx.conf
├── README.md
└── templates/
    └── index.html
```

## ❓ FAQ

**Czy muszę używać VPS?**
Nie. Z WireGuard lub w tej samej sieci możesz łączyć się bezpośrednio.

**Czy działa na CPU?**
Tak, ale wolniej.

**Czy mogę używać własnego modelu?**
Tak, zmień `MODEL_NAME` na inny ViViT z Hugging Face.

**Jak sprawdzić IP w WireGuard?**
- Windows: `ipconfig`
- Linux: `ip addr show wg0` lub `wg show`
- Android: aplikacja WireGuard → „Addresses”

**Wsparcie**
Sprawdź logi serwera i konsolę przeglądarki (F12).
