/**
 * ASAFI - Autonomous Spacecraft Avionics Fire Interceptor
 * Streamlined Real-Time Video & Telemetry Client Engine
 */

(function () {
  'use strict';

  // DOM Elements
  const dom = {
    liveVideo: document.getElementById('liveVideo'),
    streamPlaceholder: document.getElementById('streamPlaceholder'),
    linkStatusBadge: document.getElementById('linkStatusBadge'),
    linkStatusText: document.getElementById('linkStatusText'),
    cameraStatusText: document.getElementById('cameraStatusText'),
    gimbalModeText: document.getElementById('gimbalModeText'),
    thermalModeText: document.getElementById('thermalModeText'),

    targetDot: document.getElementById('targetDot'),
    targetStatusText: document.getElementById('targetStatusText'),
    targetConfText: document.getElementById('targetConfText'),
    targetClassText: document.getElementById('targetClassText'),
    fpsVal: document.getElementById('fpsVal'),
    latencyVal: document.getElementById('latencyVal'),
    centroidVal: document.getElementById('centroidVal'),
    offsetVal: document.getElementById('offsetVal'),
    gimbalAnglesVal: document.getElementById('gimbalAnglesVal'),
    thermalTempVal: document.getElementById('thermalTempVal')
  };

  let ws = null;
  let reconnectTimer = null;
  let firstFrameReceived = false;

  function getWsUrl() {
    const loc = window.location;
    const protocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = loc.host || 'localhost:8000';
    return `${protocol}//${host}/ws`;
  }

  function connect() {
    if (ws) {
      try { ws.close(); } catch (e) {}
      ws = null;
    }

    const wsUrl = getWsUrl();
    console.log('[ASAFI] Connecting to WebSocket:', wsUrl);
    setConnectionState('connecting');

    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      console.warn('[ASAFI] WebSocket construction failed:', e);
      scheduleReconnect();
      return;
    }

    ws.onopen = () => {
      console.log('[ASAFI] WebSocket connection established.');
      setConnectionState('online');
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'FRAME_UPDATE') {
          handleFrameUpdate(msg);
        }
      } catch (err) {
        console.error('[ASAFI] Error parsing WebSocket message:', err);
      }
    };

    ws.onclose = () => {
      console.warn('[ASAFI] WebSocket connection closed.');
      setConnectionState('disconnected');
      scheduleReconnect();
    };

    ws.onerror = (err) => {
      console.error('[ASAFI] WebSocket error:', err);
      setConnectionState('disconnected');
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, 2000);
  }

  function setConnectionState(state) {
    if (state === 'online') {
      dom.linkStatusBadge.className = 'hud-badge status-online';
      dom.linkStatusText.textContent = 'ONLINE';
    } else if (state === 'connecting') {
      dom.linkStatusBadge.className = 'hud-badge';
      dom.linkStatusText.textContent = 'CONNECTING...';
    } else {
      dom.linkStatusBadge.className = 'hud-badge status-disconnected';
      dom.linkStatusText.textContent = 'OFFLINE';
    }
  }

  function handleFrameUpdate(data) {
    // 1. Update Video Frame
    if (data.frame) {
      dom.liveVideo.src = data.frame;
      if (!firstFrameReceived) {
        firstFrameReceived = true;
        if (dom.streamPlaceholder) {
          dom.streamPlaceholder.classList.add('hidden');
        }
      }
    }

    // 2. Performance Metrics
    dom.fpsVal.textContent = data.fps !== undefined ? Number(data.fps).toFixed(1) : '--';
    dom.latencyVal.textContent = data.inference_ms !== undefined ? `${Number(data.inference_ms).toFixed(1)} ms` : '-- ms';

    // 3. Detection Telemetry
    const targetDetected = data.target_detected;
    const primary = data.primary_target;

    if (targetDetected && primary) {
      dom.targetDot.classList.add('active');
      dom.targetStatusText.textContent = `TARGET LOCKED (${primary.class_name.toUpperCase()})`;
      dom.targetStatusText.className = 'target-status-label fire-locked';

      const confPct = Math.round(primary.confidence * 100);
      dom.targetConfText.textContent = `${confPct}%`;
      dom.targetClassText.textContent = primary.class_name.toUpperCase();

      dom.centroidVal.textContent = `[X: ${primary.cx}, Y: ${primary.cy}]`;
    } else {
      dom.targetDot.classList.remove('active');
      dom.targetStatusText.textContent = 'OPTICAL SEARCHING...';
      dom.targetStatusText.className = 'target-status-label';
      dom.targetConfText.textContent = '0.0%';
      dom.targetClassText.textContent = 'NONE';
      dom.centroidVal.textContent = '[X: ---, Y: ---]';
    }

    // 4. Hardware & Actuation Telemetry
    if (data.telemetry) {
      const hw = data.telemetry;

      // Gimbal
      if (hw.gimbal) {
        const pan = Number(hw.gimbal.pan || 0).toFixed(1);
        const tilt = Number(hw.gimbal.tilt || 0).toFixed(1);
        dom.gimbalAnglesVal.textContent = `PAN: ${pan >= 0 ? '+' : ''}${pan}° | TILT: ${tilt >= 0 ? '+' : ''}${tilt}°`;

        if (hw.gimbal.dx !== undefined && hw.gimbal.dy !== undefined) {
          const dx = Number(hw.gimbal.dx).toFixed(1);
          const dy = Number(hw.gimbal.dy).toFixed(1);
          dom.offsetVal.textContent = `[ΔX: ${dx >= 0 ? '+' : ''}${dx}, ΔY: ${dy >= 0 ? '+' : ''}${dy}]`;
        }
      }

      // Thermal Sensor
      if (hw.thermal) {
        const obj = Number(hw.thermal.object_c || 24.5).toFixed(1);
        const amb = Number(hw.thermal.ambient_c || 24.0).toFixed(1);
        dom.thermalTempVal.textContent = `OBJ: ${obj}°C | AMB: ${amb}°C`;
        if (hw.thermal.fire_confirmed) {
          dom.thermalTempVal.className = 'val text-red';
        } else {
          dom.thermalTempVal.className = 'val text-green';
        }
      }

      // Hardware Badges
      if (hw.hardware_mode) {
        dom.gimbalModeText.textContent = hw.hardware_mode.gimbal || 'MOCK';
        dom.gimbalModeText.className = hw.hardware_mode.gimbal === 'PHYSICAL' ? 'badge-val text-green' : 'badge-val text-amber';

        dom.thermalModeText.textContent = hw.hardware_mode.thermal || 'MOCK';
        dom.thermalModeText.className = hw.hardware_mode.thermal === 'PHYSICAL' ? 'badge-val text-green' : 'badge-val text-amber';
      }
    }
  }

  // Auto-connect on page load
  window.addEventListener('DOMContentLoaded', () => {
    connect();
  });
})();
