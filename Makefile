PY := .venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help run debug no-tui list fake fake-grab autoplay maxautoplay dashboard-only sniff install

help:  ## Show this help
	@awk 'BEGIN{FS=":.*?## "} /^[a-zA-Z_-]+:.*?## /{printf "  %-12s %s\n",$$1,$$2}' $(MAKEFILE_LIST)

run:       ## Launch synth with curses TUI
	$(PY) theremin_wind.py --dashboard --dmx

debug:     ## Dump MIDI bytes (no TUI)
	$(PY) theremin_wind.py --dashboard --dmx --debug

no-tui:    ## Play without TUI
	$(PY) theremin_wind.py --dashboard --dmx --no-tui

list:      ## List serial + audio devices
	$(PY) theremin_wind.py --list

fake:      ## Drive synth from a touchpad
	$(PY) theremin_wind.py --dashboard --dmx --fake

fake-grab: ## Touchpad mode with exclusive grab
	$(PY) theremin_wind.py --dashboard --dmx --fake --grab

autoplay:  ## Demo mode: synth plays itself, no theremin
	$(PY) theremin_wind.py --dashboard --dmx --autoplay

maxautoplay: ## Demo mode at full volume, only pitch moves
	$(PY) theremin_wind.py --dashboard --dmx --maxautoplay

dashboard-only: ## Autoplay the dashboard with no audio (silent kiosk signal)
	$(PY) theremin_wind.py --dashboard --autoplay --mute

sniff:     ## Run serial baud-rate diagnostic
	$(PY) sniff_serial.py

install:   ## Create venv and install requirements
	uv venv .venv
	uv pip install --python .venv/bin/python -r requirements.txt
