# Box-in-bag detection

Rule-based detector for a person putting a box into a bag, or opening a box. YOLO finds person / box / bag / open box, then state machines decide whether the action happened.

## Setup

```bash
pip install -r requirements.txt
cp config.local.yaml.example config.local.yaml   # optional; or just edit config.local.yaml
python3 run.py
```

`config.yaml` is safe to commit. Put camera credentials in `config.local.yaml` (gitignored).

```yaml
source:
  path: rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101/
```

## Run

```bash
python3 run.py                         # uses config (and config.local.yaml if present)
python3 run.py --source video.mp4
python3 run.py --test-alert            # desktop + sound alert without the camera
```

In the preview window: click **REC** or press **R** to record, **q** to quit.

## Outputs

| Path | Contents |
|---|---|
| `outputs/alerts/` | JPEG snapshots on each event |
| `outputs/recordings/events/` | Clips around each put-box-in-bag event |
| `outputs/recordings/manual/` | Takes from the REC button |

Alerts also fire a macOS notification. Set `alerts.webhook_url` for Slack / Discord / n8n.
