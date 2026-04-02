# Real-time Video Captioning

System do analizy wideo w czasie rzeczywistym z wykorzystaniem modeli RT-DETR, ViViT, IntenVL. Źródłem obrazu jest kamera urządzenia, na którym otwarta jest strona (telefon/komputer).

## 🎯 Architektura

```
Telefon (Android) → Kamera → Przeglądarka → Komputer z CUDA  → Wynik → Wyświetlanie
```

## 📋 Wymagania

### Komputer z CUDA:
- Python 3.8+
- NVIDIA GPU
- Zainstalowany CUDA Toolkit

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

## 🔒 Bezpieczeństwo

**WireGuard:**
- Ruch szyfrowany przez tunel VPN; HTTPS nie wymagane

**HTTPS lokalnie na Windows**

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
$env:SSL_CERT_PATH = localhost.pem"
$env:SSL_KEY_PATH  = "localhost-key.pem"
python main.py
```

4) Otwórz stronę:
- Na tym komputerze: `https://localhost:5000`
- Z telefonu: `https://IP_WIREGUARD:5000` lub `https://IP_LAN:5000`

## 📞 Wydajność

**GPU (RTX 3060+):**
- Ładowanie modelu: ~5–10 s
- Inference: ~0.5 s/predykcja

**CPU:**
- Ładowanie modelu: ~10–20 s
- Inference: ~3–5 s/predykcja
