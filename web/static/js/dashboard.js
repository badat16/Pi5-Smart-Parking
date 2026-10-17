/**
 * Realtime JavaScript Client for Smart Parking Web Dashboard (Raspberry Pi 5)
 * Polling interval: 500ms
 */

document.addEventListener("DOMContentLoaded", () => {
    // Start WebRTC ultra-low latency streams
    initWebRTCStream("tracking", "video-tracking", "stream-tracking", "tag-proto-tracking");
    initWebRTCStream("gate", "video-gate", "stream-gate", "tag-proto-gate");

    // Start initial polling loops
    fetchSystemInfo();
    fetchParkingInfo();
    fetchStatusInfo();
    fetchRecentSessions();

    setInterval(fetchSystemInfo, 1000);
    setInterval(fetchParkingInfo, 500);
    setInterval(fetchStatusInfo, 500);
    setInterval(fetchRecentSessions, 3000);
});

async function fetchSystemInfo() {
    try {
        const res = await fetch("/api/system");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("val-fps").innerText = data.fps || "0";
        document.getElementById("val-cpu").innerText = `${data.cpu_percent}%`;
        document.getElementById("val-ram").innerText = `${data.ram_percent}%`;
        
        if (data.cpu_temp !== null && data.cpu_temp !== undefined) {
            document.getElementById("val-temp").innerText = `${data.cpu_temp} °C`;
        } else {
            document.getElementById("val-temp").innerText = "N/A";
        }

        // Camera indicators
        const indTrack = document.getElementById("ind-cam-track");
        const indGate = document.getElementById("ind-cam-gate");

        if (data.camera_status.tracking_camera === "ONLINE") {
            indTrack.querySelector(".dot").style.backgroundColor = "var(--accent-green)";
        } else {
            indTrack.querySelector(".dot").style.backgroundColor = "var(--accent-red)";
        }

        if (data.camera_status.gate_camera === "ONLINE") {
            indGate.querySelector(".dot").style.backgroundColor = "var(--accent-green)";
        } else {
            indGate.querySelector(".dot").style.backgroundColor = "var(--accent-red)";
        }
    } catch (err) {
        console.warn("System telemetry fetch error:", err);
    }
}

async function fetchParkingInfo() {
    try {
        const res = await fetch("/api/parking");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("sum-occupied").innerText = data.total_occupied || 0;
        document.getElementById("sum-vacant").innerText = data.total_vacant || 0;
        document.getElementById("sum-pending").innerText = data.pending_count || 0;

        // Render 12 slot cards
        if (data.slots) {
            Object.keys(data.slots).forEach(slotId => {
                const slotData = data.slots[slotId];
                const cardEl = document.getElementById(`slot-${slotId}`);
                if (!cardEl) return;

                const statusEl = cardEl.querySelector(".slot-status");
                const infoEl = cardEl.querySelector(".slot-info");

                if (slotData.occupied) {
                    cardEl.classList.remove("vacant");
                    cardEl.classList.add("occupied");
                    statusEl.innerText = "ĐANG ĐỖ";
                    
                    let infoTxt = `Xe #${slotData.vehicle_id || '?'}`;
                    infoEl.innerText = infoTxt;
                } else {
                    cardEl.classList.remove("occupied");
                    cardEl.classList.add("vacant");
                    statusEl.innerText = "TRỐNG";
                    infoEl.innerText = "--";
                }
            });
        }
    } catch (err) {
        console.warn("Parking occupancy fetch error:", err);
    }
}

async function fetchStatusInfo() {
    try {
        const res = await fetch("/api/status");
        if (!res.ok) return;
        const data = await res.json();

        // System status banner
        const bannerEl = document.getElementById("system-status-banner");
        bannerEl.querySelector(".banner-title").innerText = data.status_msg || "SẴN SÀNG";
        bannerEl.querySelector(".banner-sub").innerText = data.last_transaction || "Chưa có giao dịch";
        bannerEl.style.borderLeftColor = data.status_color_hex || "var(--accent-blue)";

        // Barrier pill
        const barrierPill = document.getElementById("barrier-status-pill");
        barrierPill.innerText = data.barrier_status || "BARRIER: CLOSED";
        if (data.barrier_status && data.barrier_status.includes("OPEN")) {
            barrierPill.classList.add("open");
        } else {
            barrierPill.classList.remove("open");
        }
    } catch (err) {
        console.warn("Status fetch error:", err);
    }
}

async function fetchRecentSessions() {
    try {
        const res = await fetch("/api/sessions");
        if (!res.ok) return;
        const data = await res.json();

        const tbody = document.getElementById("sessions-table-body");
        if (!data.recent_sessions || data.recent_sessions.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="empty-msg">Chưa có thông tin phiên đỗ xe...</td></tr>`;
            return;
        }

        let html = "";
        data.recent_sessions.forEach(sess => {
            const statusBadge = sess.active == 1 
                ? `<span class="badge-active text-success">● DANG DO</span>`
                : `<span class="badge-closed text-secondary">✓ DA RA</span>`;

            html += `
                <tr>
                    <td>#${sess.id}</td>
                    <td><strong class="text-accent">${sess.plate}</strong></td>
                    <td>${sess.person_name || 'Guest'}</td>
                    <td>${sess.time_in || '--'}</td>
                    <td>${sess.time_out || '--'}</td>
                    <td>${sess.duration_minutes ? sess.duration_minutes + ' phút' : '--'}</td>
                    <td>${statusBadge}</td>
                </tr>
            `;
        });
        tbody.innerHTML = html;
    } catch (err) {
        console.warn("Sessions history fetch error:", err);
    }
}

async function triggerAction(actionName) {
    try {
        const res = await fetch(`/api/action/${actionName}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        const data = await res.json();
        if (data.success) {
            console.log(`Action '${actionName}' triggered successfully.`);
        } else {
            alert(`Lỗi kích hoạt hành động: ${data.error}`);
        }
    } catch (err) {
        alert(`Không thể kết nối đến máy chủ: ${err}`);
    }
}

// -------------------------------------------------------------
// WebRTC Stream Management (Ultra-Low Latency UDP with MJPEG Fallback)
// -------------------------------------------------------------
const peerConnections = {};

async function initWebRTCStream(streamName, videoId, imgId, badgeId) {
    const videoEl = document.getElementById(videoId);
    const imgEl = document.getElementById(imgId);
    const badgeEl = document.getElementById(badgeId);

    function fallbackToMJPEG(reason) {
        console.warn(`[WebRTC] Falling back to MJPEG for ${streamName}: ${reason}`);
        if (badgeEl) {
            badgeEl.innerText = "MJPEG TCP";
            badgeEl.classList.add("mjpeg");
        }
        if (videoEl) videoEl.style.display = "none";
        if (imgEl) {
            imgEl.style.display = "block";
            imgEl.src = `/video_feed/${streamName}?t=` + Date.now();
        }
    }

    if (!window.RTCPeerConnection) {
        fallbackToMJPEG("Browser lacks RTCPeerConnection support");
        return;
    }

    try {
        if (peerConnections[streamName]) {
            try { peerConnections[streamName].close(); } catch (e) {}
            delete peerConnections[streamName];
        }

        const pc = new RTCPeerConnection({
            iceServers: [
                { urls: "stun:stun.l.google.com:19302" }
            ]
        });
        peerConnections[streamName] = pc;

        // Indicate that client only wants to receive video
        pc.addTransceiver("video", { direction: "recvonly" });

        pc.ontrack = (event) => {
            if (event.track.kind === "video") {
                const stream = event.streams && event.streams[0] ? event.streams[0] : new MediaStream([event.track]);
                videoEl.srcObject = stream;
                videoEl.style.display = "block";
                if (imgEl) imgEl.style.display = "none";
                if (badgeEl) {
                    badgeEl.innerText = "WebRTC UDP";
                    badgeEl.classList.remove("mjpeg");
                }
                videoEl.play().catch(e => console.log("Video auto-play policy suppressed:", e));
            }
        };

        pc.oniceconnectionstatechange = () => {
            if (pc.iceConnectionState === "failed") {
                fallbackToMJPEG("ICE connection state: failed");
            }
        };

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        // Wait for ICE gathering to collect local host candidates
        await new Promise((resolve) => {
            if (pc.iceGatheringState === "complete") {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === "complete") {
                        pc.removeEventListener("icegatheringstatechange", checkState);
                        resolve();
                    }
                };
                pc.addEventListener("icegatheringstatechange", checkState);
                setTimeout(resolve, 800); // 800ms safety timeout
            }
        });

        const res = await fetch(`/webrtc/offer?stream=${streamName}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sdp: pc.localDescription.sdp,
                type: pc.localDescription.type,
                stream: streamName
            })
        });

        if (!res.ok) {
            throw new Error(`Signaling error (HTTP ${res.status})`);
        }

        const answer = await res.json();
        if (answer.error) {
            throw new Error(answer.error);
        }

        await pc.setRemoteDescription(new RTCSessionDescription(answer));

    } catch (err) {
        fallbackToMJPEG(err.message || err);
    }
}

// Cleanly close WebRTC sessions when leaving the dashboard tab
window.addEventListener("beforeunload", () => {
    Object.values(peerConnections).forEach(pc => {
        try { pc.close(); } catch (e) {}
    });
});

