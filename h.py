'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SolarScan AI — Solar Panel Dust Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
<style>
/* ============================================================
   RESET & BASE
   ============================================================ */
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg:        #050B16;
  --bg2:       #080F1F;
  --card:      rgba(255,255,255,0.05);
  --card-hov:  rgba(255,255,255,0.09);
  --border:    rgba(255,255,255,0.08);
  --border2:   rgba(255,255,255,0.14);
  --primary:   #00D084;
  --primary-d: #00a86a;
  --secondary: #1E90FF;
  --accent:    #FFD700;
  --text:      #F0F4FF;
  --muted:     #8896AB;
  --danger:    #FF4D6A;
  --warn:      #FFB020;
  --glass:     rgba(8,15,31,0.72);
}

html { scroll-behavior: smooth; font-size: 16px; }

body {
  font-family: 'Poppins', sans-serif;
  background: var(--bg);
  color: var(--text);
  overflow-x: hidden;
  line-height: 1.7;
}

/* ============================================================
   CANVAS PARTICLES
   ============================================================ */
#particle-canvas {
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
}

/* ============================================================
   UTILITY
   ============================================================ */
.container { max-width: 1160px; margin: 0 auto; padding: 0 24px; }
.section    { padding: 100px 0; position: relative; z-index: 1; }
.section-label {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--primary);
  background: rgba(0,208,132,0.1);
  border: 1px solid rgba(0,208,132,0.25);
  padding: 5px 14px;
  border-radius: 20px;
  margin-bottom: 16px;
}
.section-heading {
  font-size: clamp(1.7rem, 3.5vw, 2.6rem);
  font-weight: 700;
  line-height: 1.22;
  margin-bottom: 16px;
}
.section-sub {
  font-size: 1rem;
  color: var(--muted);
  max-width: 540px;
  line-height: 1.75;
}
.center { text-align: center; }
.center .section-sub { margin: 0 auto; }

/* BUTTONS */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 13px 30px;
  border-radius: 50px;
  font-family: 'Poppins', sans-serif;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  border: none;
  transition: transform 0.22s, box-shadow 0.22s, opacity 0.18s;
  text-decoration: none;
}
.btn:hover { transform: translateY(-2px); opacity: 0.92; }
.btn-primary {
  background: linear-gradient(135deg, var(--primary), #00b870);
  color: #030a0e;
  box-shadow: 0 0 28px rgba(0,208,132,0.3);
}
.btn-primary:hover { box-shadow: 0 0 44px rgba(0,208,132,0.5); }
.btn-outline {
  background: transparent;
  color: var(--text);
  border: 1.5px solid var(--border2);
  backdrop-filter: blur(8px);
}
.btn-outline:hover { border-color: var(--primary); color: var(--primary); }

/* GLASS CARD */
.glass {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 18px;
  backdrop-filter: blur(12px);
  transition: background 0.3s, border-color 0.3s, transform 0.3s, box-shadow 0.3s;
}
.glass:hover {
  background: var(--card-hov);
  border-color: var(--border2);
  transform: translateY(-4px);
  box-shadow: 0 16px 48px rgba(0,0,0,0.35);
}

/* FADE-IN ANIMATION */
.reveal {
  opacity: 0;
  transform: translateY(36px);
  transition: opacity 0.7s ease, transform 0.7s ease;
}
.reveal.visible { opacity: 1; transform: translateY(0); }

/* Loading Spinner */
.loader {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255,255,255,0.3);
  border-radius: 50%;
  border-top-color: var(--primary);
  animation: spin 0.8s linear infinite;
  margin-right: 8px;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Result Container */
.result-container {
  margin-top: 30px;
  padding: 25px;
  background: rgba(0,208,132,0.08);
  border: 1px solid rgba(0,208,132,0.2);
  border-radius: 20px;
  backdrop-filter: blur(10px);
  animation: fadeIn 0.5s ease;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}

.result-badge {
  font-size: 3rem;
  margin-bottom: 10px;
}
.result-title {
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 20px;
}
.result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 20px;
  margin-top: 20px;
}
.result-card {
  background: rgba(255,255,255,0.05);
  padding: 15px;
  border-radius: 12px;
  text-align: center;
}
.result-card-label {
  font-size: 0.75rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 5px;
}
.result-card-value {
  font-size: 1.4rem;
  font-weight: 700;
  color: var(--primary);
}

/* ============================================================
   NAVBAR
   ============================================================ */
nav {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 999;
  padding: 16px 0;
  transition: background 0.4s, border-bottom 0.4s;
}
nav.scrolled {
  background: rgba(5,11,22,0.92);
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(16px);
}
.nav-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.logo {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 1.12rem;
  font-weight: 700;
  color: var(--text);
  text-decoration: none;
}
.logo-icon {
  width: 34px; height: 34px;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.9rem;
}
.logo span { color: var(--primary); }

.nav-links {
  display: flex;
  align-items: center;
  gap: 4px;
  list-style: none;
}
.nav-links a {
  color: var(--muted);
  text-decoration: none;
  font-size: 0.88rem;
  font-weight: 500;
  padding: 7px 14px;
  border-radius: 8px;
  transition: color 0.2s, background 0.2s;
}
.nav-links a:hover { color: var(--text); background: rgba(255,255,255,0.06); }

.nav-cta { margin-left: 12px; }
.hamburger { display: none; background: none; border: none; cursor: pointer; color: var(--text); font-size: 1.4rem; }

.mobile-menu {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(5,11,22,0.97);
  z-index: 998;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 28px;
}
.mobile-menu.open { display: flex; }
.mobile-menu a {
  color: var(--text);
  text-decoration: none;
  font-size: 1.5rem;
  font-weight: 600;
  transition: color 0.2s;
}
.mobile-menu a:hover { color: var(--primary); }
.mobile-close { position: absolute; top: 24px; right: 28px; font-size: 1.6rem; background: none; border: none; color: var(--muted); cursor: pointer; }

/* ============================================================
   HERO
   ============================================================ */
#hero {
  min-height: 100vh;
  display: flex;
  align-items: center;
  position: relative;
  overflow: hidden;
  padding-top: 90px;
}
.hero-bg {
  position: absolute;
  inset: 0;
  background:
    linear-gradient(180deg, rgba(0,208,132,0.05) 0%, transparent 50%),
    radial-gradient(ellipse 70% 60% at 50% 0%, rgba(30,144,255,0.1) 0%, transparent 70%),
    url("https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1920&q=60")
    center / cover no-repeat;
  filter: brightness(0.25);
  z-index: 0;
}
.hero-content {
  position: relative;
  z-index: 1;
  max-width: 740px;
}
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: rgba(0,208,132,0.1);
  border: 1px solid rgba(0,208,132,0.3);
  padding: 6px 16px;
  border-radius: 30px;
  font-size: 0.8rem;
  color: var(--primary);
  font-weight: 500;
  margin-bottom: 24px;
  animation: pulse-badge 3s infinite;
}
@keyframes pulse-badge {
  0%,100% { box-shadow: 0 0 0 0 rgba(0,208,132,0.3); }
  50%      { box-shadow: 0 0 0 8px rgba(0,208,132,0); }
}
.hero h1 {
  font-size: clamp(2.2rem, 5.5vw, 3.8rem);
  font-weight: 800;
  line-height: 1.12;
  margin-bottom: 22px;
  letter-spacing: -0.01em;
}
.hero h1 .highlight {
  background: linear-gradient(90deg, var(--primary), var(--secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 1.08rem;
  color: var(--muted);
  max-width: 560px;
  margin-bottom: 36px;
  line-height: 1.8;
}
.hero-buttons { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 56px; }
.hero-stats {
  display: flex;
  gap: 36px;
  flex-wrap: wrap;
}
.stat-item { display: flex; flex-direction: column; }
.stat-value { font-size: 1.7rem; font-weight: 700; color: var(--text); }
.stat-value span { color: var(--primary); }
.stat-label { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

/* Floating orbs */
.orb {
  position: absolute;
  border-radius: 50%;
  pointer-events: none;
  animation: float linear infinite;
}
.orb-1 { width: 380px; height: 380px; right: -60px; top: 10%; background: radial-gradient(circle, rgba(0,208,132,0.08), transparent 70%); animation-duration: 18s; }
.orb-2 { width: 260px; height: 260px; right: 120px; bottom: 10%; background: radial-gradient(circle, rgba(30,144,255,0.08), transparent 70%); animation-duration: 24s; }
@keyframes float {
  0%,100% { transform: translateY(0) scale(1); }
  50%      { transform: translateY(-30px) scale(1.04); }
}

/* ============================================================
   ABOUT
   ============================================================ */
#about { background: var(--bg2); }
.about-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 60px;
  align-items: center;
}
.about-img-wrap {
  position: relative;
  border-radius: 22px;
  overflow: hidden;
}
.about-img-wrap img {
  width: 100%; display: block; border-radius: 22px;
  filter: brightness(0.85) saturate(1.1);
}
.about-img-badge {
  position: absolute;
  bottom: 20px; left: 20px;
  background: rgba(5,11,22,0.85);
  border: 1px solid var(--border2);
  border-radius: 12px;
  padding: 12px 18px;
  backdrop-filter: blur(10px);
}
.about-img-badge .val { font-size: 1.4rem; font-weight: 700; color: var(--primary); }
.about-img-badge .lbl { font-size: 0.72rem; color: var(--muted); }
.about-list { list-style: none; margin-top: 24px; display: flex; flex-direction: column; gap: 12px; }
.about-list li {
  display: flex; align-items: flex-start; gap: 10px;
  font-size: 0.92rem; color: var(--muted);
}
.about-list li i { color: var(--primary); margin-top: 3px; font-size: 0.85rem; flex-shrink: 0; }

/* ============================================================
   FEATURES
   ============================================================ */
.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 22px;
  margin-top: 52px;
}
.feat-card {
  padding: 32px 28px;
  position: relative;
  overflow: hidden;
}
.feat-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--primary), transparent);
  opacity: 0;
  transition: opacity 0.3s;
}
.feat-card:hover::before { opacity: 1; }
.feat-icon {
  width: 52px; height: 52px;
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem;
  margin-bottom: 20px;
}
.feat-card h3 { font-size: 1.05rem; font-weight: 600; margin-bottom: 10px; }
.feat-card p  { font-size: 0.87rem; color: var(--muted); line-height: 1.75; }

/* icon color variants */
.icon-green  { background: rgba(0,208,132,0.12); color: var(--primary); }
.icon-blue   { background: rgba(30,144,255,0.12); color: var(--secondary); }
.icon-gold   { background: rgba(255,215,0,0.12);  color: var(--accent); }
.icon-red    { background: rgba(255,77,106,0.12); color: var(--danger); }
.icon-orange { background: rgba(255,176,32,0.12); color: var(--warn); }
.icon-purple { background: rgba(156,100,255,0.12); color: #9c64ff; }

/* ============================================================
   HOW IT WORKS — TIMELINE
   ============================================================ */
#how { background: var(--bg2); }
.timeline {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 0;
  margin-top: 60px;
  position: relative;
  flex-wrap: wrap;
}
.tl-step {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  max-width: 170px;
  position: relative;
  flex: 1;
  min-width: 130px;
}
.tl-connector {
  flex: 1;
  height: 2px;
  background: linear-gradient(90deg, var(--primary), var(--secondary));
  margin-top: 30px;
  min-width: 20px;
  opacity: 0.35;
}
.tl-circle {
  width: 60px; height: 60px;
  border-radius: 50%;
  background: linear-gradient(135deg, rgba(0,208,132,0.15), rgba(30,144,255,0.15));
  border: 2px solid rgba(0,208,132,0.35);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.25rem;
  margin-bottom: 16px;
  position: relative;
  transition: transform 0.3s, box-shadow 0.3s;
}
.tl-step:hover .tl-circle {
  transform: scale(1.12);
  box-shadow: 0 0 24px rgba(0,208,132,0.35);
}
.tl-num {
  position: absolute;
  top: -8px; right: -8px;
  width: 20px; height: 20px;
  background: var(--primary);
  color: #030a0e;
  font-size: 0.65rem;
  font-weight: 700;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
}
.tl-step h4 { font-size: 0.88rem; font-weight: 600; margin-bottom: 6px; }
.tl-step p  { font-size: 0.75rem; color: var(--muted); line-height: 1.6; }

/* ============================================================
   STATS
   ============================================================ */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 22px;
  margin-top: 52px;
}
.stat-card {
  padding: 32px 24px;
  text-align: center;
}
.stat-card .s-icon {
  font-size: 2rem;
  margin-bottom: 14px;
  display: block;
}
.stat-card .s-val {
  font-size: 2.2rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
  margin-bottom: 8px;
}
.stat-card .s-label { font-size: 0.85rem; color: var(--muted); }

/* ============================================================
   DASHBOARD PREVIEW
   ============================================================ */
#dashboard { background: var(--bg2); overflow: hidden; }
.dash-wrap {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border2);
  border-radius: 22px;
  overflow: hidden;
  margin-top: 52px;
  box-shadow: 0 24px 80px rgba(0,0,0,0.45);
}
.dash-topbar {
  background: rgba(255,255,255,0.04);
  border-bottom: 1px solid var(--border);
  padding: 14px 20px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.dash-dot { width: 12px; height: 12px; border-radius: 50%; }
.d1 { background: #ff5f57; }
.d2 { background: #febc2e; }
.d3 { background: #28c840; }
.dash-title { font-size: 0.78rem; color: var(--muted); margin-left: 8px; }

.dash-body {
  display: grid;
  grid-template-columns: 1.2fr 1fr;
  gap: 0;
}
.dash-left {
  padding: 28px;
  border-right: 1px solid var(--border);
}
.dash-right { padding: 28px; }

.dash-panel-img {
  width: 100%;
  height: 180px;
  background:
    linear-gradient(135deg, rgba(0,208,132,0.08), rgba(30,144,255,0.06)),
    url("https://images.unsplash.com/photo-1560472354-b33ff0c44a43?auto=format&fit=crop&w=600&q=60")
    center/cover;
  border-radius: 12px;
  margin-bottom: 18px;
  position: relative;
  overflow: hidden;
}
.dust-overlay {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 60% 40% at 65% 45%, rgba(255,215,0,0.25) 0%, transparent 60%),
    radial-gradient(ellipse 40% 30% at 30% 60%, rgba(255,130,0,0.15) 0%, transparent 55%);
}
.scan-line {
  position: absolute;
  left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--primary), transparent);
  animation: scan 2.5s linear infinite;
  opacity: 0.7;
}
@keyframes scan { 0% { top: 0; } 100% { top: 100%; } }
.label-chip {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 6px;
  font-size: 0.68rem;
  font-weight: 600;
}
.chip-warn  { background: rgba(255,176,32,0.15); color: var(--warn); }
.chip-ok    { background: rgba(0,208,132,0.12); color: var(--primary); }
.chip-bad   { background: rgba(255,77,106,0.12); color: var(--danger); }

.dash-metric-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.dash-metric-label { font-size: 0.78rem; color: var(--muted); }
.dash-metric-val   { font-size: 0.9rem; font-weight: 600; }

.progress-bar-wrap { margin-bottom: 16px; }
.pb-label { display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--muted); margin-bottom: 5px; }
.pb-track { height: 7px; background: rgba(255,255,255,0.07); border-radius: 4px; overflow: hidden; }
.pb-fill  { height: 100%; border-radius: 4px; transition: width 1s ease; }
.fill-green  { background: linear-gradient(90deg, var(--primary), #00b870); }
.fill-warn   { background: linear-gradient(90deg, var(--warn), #ffca50); }
.fill-danger { background: linear-gradient(90deg, var(--danger), #ff7a8a); }

/* Donut chart */
.donut-wrap { display: flex; justify-content: center; margin: 18px 0; }
.donut { position: relative; width: 120px; height: 120px; }
.donut svg { transform: rotate(-90deg); }
.donut-text {
  position: absolute;
  inset: 0; display: flex; flex-direction: column;
  align-items: center; justify-content: center;
}
.donut-text .d-val { font-size: 1.4rem; font-weight: 700; line-height: 1; color: var(--warn); }
.donut-text .d-lbl { font-size: 0.68rem; color: var(--muted); margin-top: 2px; }

/* ============================================================
   SAMPLE RESULTS
   ============================================================ */
.results-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 22px;
  margin-top: 52px;
}
.res-card {
  padding: 28px;
  text-align: center;
}
.res-icon { font-size: 2.4rem; margin-bottom: 14px; display: block; }
.res-title { font-size: 1rem; font-weight: 600; margin-bottom: 4px; }
.res-dust { font-size: 0.8rem; color: var(--muted); margin-bottom: 18px; }
.res-bar-wrap { height: 10px; background: rgba(255,255,255,0.07); border-radius: 5px; overflow: hidden; margin-bottom: 14px; }
.res-bar { height: 100%; border-radius: 5px; }
.res-status {
  font-size: 0.78rem; font-weight: 600; padding: 5px 14px;
  border-radius: 20px; display: inline-block;
}

/* ============================================================
   BENEFITS
   ============================================================ */
#benefits { background: var(--bg2); }
.benefits-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 28px;
  margin-top: 52px;
}
.ben-card {
  padding: 36px 28px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.ben-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--primary), var(--secondary));
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 0.4s;
}
.ben-card:hover::after { transform: scaleX(1); }
.ben-icon { font-size: 2.2rem; color: var(--primary); margin-bottom: 18px; display: block; }
.ben-card h3 { font-size: 1.05rem; font-weight: 600; margin-bottom: 10px; }
.ben-card p  { font-size: 0.85rem; color: var(--muted); line-height: 1.75; }

/* ============================================================
   FAQ
   ============================================================ */
.faq-list { max-width: 760px; margin: 52px auto 0; display: flex; flex-direction: column; gap: 12px; }
.faq-item {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}
.faq-q {
  width: 100%; text-align: left; background: none; border: none;
  color: var(--text);
  font-family: 'Poppins', sans-serif;
  font-size: 0.95rem; font-weight: 500;
  padding: 20px 24px;
  cursor: pointer;
  display: flex; align-items: center; justify-content: space-between;
  gap: 12px;
  transition: color 0.2s;
}
.faq-q:hover { color: var(--primary); }
.faq-icon { font-size: 0.75rem; color: var(--muted); transition: transform 0.35s; flex-shrink: 0; }
.faq-item.open .faq-icon { transform: rotate(180deg); color: var(--primary); }
.faq-a {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.42s ease, padding 0.3s;
  font-size: 0.88rem;
  color: var(--muted);
  line-height: 1.8;
  padding: 0 24px;
}
.faq-item.open .faq-a { max-height: 200px; padding: 0 24px 20px; }

/* ============================================================
   CTA
   ============================================================ */
#cta {
  position: relative;
  overflow: hidden;
  text-align: center;
  padding: 120px 0;
}
.cta-bg {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 70% at 50% 50%, rgba(0,208,132,0.09) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 30% 50%, rgba(30,144,255,0.07) 0%, transparent 60%);
}
.cta-glow {
  position: absolute;
  width: 500px; height: 500px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(0,208,132,0.12), transparent 70%);
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  animation: glow-pulse 4s ease-in-out infinite;
}
@keyframes glow-pulse {
  0%,100% { transform: translate(-50%,-50%) scale(1); opacity: 0.6; }
  50%      { transform: translate(-50%,-50%) scale(1.15); opacity: 1; }
}
#cta h2 { font-size: clamp(1.8rem, 4vw, 3rem); font-weight: 800; margin-bottom: 16px; position: relative; z-index: 1; }
#cta p  { color: var(--muted); margin-bottom: 36px; position: relative; z-index: 1; font-size: 1rem; }

/* ============================================================
   FOOTER
   ============================================================ */
footer {
  background: #030810;
  border-top: 1px solid var(--border);
  padding: 64px 0 32px;
}
.footer-grid {
  display: grid;
  grid-template-columns: 1.6fr 1fr 1fr 1fr;
  gap: 40px;
  margin-bottom: 48px;
}
.footer-brand p { font-size: 0.85rem; color: var(--muted); margin-top: 12px; line-height: 1.75; max-width: 240px; }
.footer-col h5 { font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text); margin-bottom: 16px; }
.footer-col ul { list-style: none; display: flex; flex-direction: column; gap: 9px; }
.footer-col ul a { color: var(--muted); text-decoration: none; font-size: 0.85rem; transition: color 0.2s; }
.footer-col ul a:hover { color: var(--primary); }
.footer-social { display: flex; gap: 10px; margin-top: 16px; }
.footer-social a {
  width: 36px; height: 36px;
  border-radius: 10px;
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  color: var(--muted);
  text-decoration: none;
  font-size: 0.85rem;
  transition: all 0.2s;
}
.footer-social a:hover { color: var(--primary); border-color: rgba(0,208,132,0.3); background: rgba(0,208,132,0.08); }
.footer-bottom {
  border-top: 1px solid var(--border);
  padding-top: 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  font-size: 0.8rem;
  color: var(--muted);
}
.footer-bottom a { color: var(--muted); text-decoration: none; transition: color 0.2s; }
.footer-bottom a:hover { color: var(--primary); }

/* ============================================================
   RESPONSIVE
   ============================================================ */
@media (max-width: 900px) {
  .nav-links, .nav-cta { display: none; }
  .hamburger { display: block; }
  .about-grid { grid-template-columns: 1fr; }
  .stats-grid { grid-template-columns: repeat(2, 1fr); }
  .results-grid { grid-template-columns: 1fr; }
  .benefits-grid { grid-template-columns: 1fr; }
  .dash-body { grid-template-columns: 1fr; }
  .dash-left { border-right: none; border-bottom: 1px solid var(--border); }
  .footer-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 580px) {
  .section { padding: 70px 0; }
  .timeline { flex-direction: column; align-items: center; }
  .tl-connector { display: none; }
  .stats-grid { grid-template-columns: 1fr 1fr; }
  .footer-grid { grid-template-columns: 1fr; }
  .hero-stats { gap: 20px; }
  .features-grid { grid-template-columns: 1fr; }
  .results-grid { grid-template-columns: 1fr; }
  .benefits-grid { grid-template-columns: 1fr; }
  .result-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<!-- ============================================================
     PARTICLE CANVAS
     ============================================================ -->
<canvas id="particle-canvas"></canvas>

<!-- ============================================================
     NAVBAR
     ============================================================ -->
<nav id="navbar">
  <div class="container nav-inner">
    <a href="#" class="logo">
      <div class="logo-icon"><i class="fas fa-sun"></i></div>
      Solar<span>Scan</span> AI
    </a>
    <ul class="nav-links">
      <li><a href="#about">About</a></li>
      <li><a href="#features">Features</a></li>
      <li><a href="#how">How It Works</a></li>
      <li><a href="#dashboard">Dashboard</a></li>
      <li><a href="#faq">FAQ</a></li>
    </ul>
    <a href="#cta" class="btn btn-primary nav-cta" style="font-size:0.85rem;padding:10px 22px">Upload Video</a>
    <button class="hamburger" id="hamburger" aria-label="Open menu"><i class="fas fa-bars"></i></button>
  </div>
</nav>

<!-- Mobile Menu -->
<div class="mobile-menu" id="mobileMenu">
  <button class="mobile-close" id="mobileClose"><i class="fas fa-times"></i></button>
  <a href="#about" onclick="closeMobile()">About</a>
  <a href="#features" onclick="closeMobile()">Features</a>
  <a href="#how" onclick="closeMobile()">How It Works</a>
  <a href="#dashboard" onclick="closeMobile()">Dashboard</a>
  <a href="#faq" onclick="closeMobile()">FAQ</a>
  <a href="#cta" class="btn btn-primary" onclick="closeMobile()">Upload Video</a>
</div>

<!-- ============================================================
     HERO
     ============================================================ -->
<section id="hero">
  <div class="hero-bg"></div>
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="container hero-content">
    <div class="hero-badge reveal">
      <i class="fas fa-circle" style="font-size:0.45rem;animation:pulse-badge 2s infinite"></i>
      AI-Powered Detection · Real-Time Analysis
    </div>
    <h1 class="reveal">
      AI-Powered<br>
      <span class="highlight">Solar Panel</span><br>
      Dust Detection
    </h1>
    <p class="hero-sub reveal">
      Detect dust accumulation, estimate energy loss, and receive intelligent cleaning recommendations
      using state-of-the-art computer vision and deep learning.
    </p>
    <div class="hero-buttons reveal">
      <a href="#cta" class="btn btn-primary"><i class="fas fa-upload"></i> Upload Video</a>
      <a href="#dashboard" class="btn btn-outline"><i class="fas fa-chart-bar"></i> View Dashboard</a>
    </div>
    <div class="hero-stats reveal">
      <div class="stat-item">
        <span class="stat-value"><span id="c1">0</span>%</span>
        <span class="stat-label">Detection Accuracy</span>
      </div>
      <div class="stat-item">
        <span class="stat-value"><span id="c2">0</span>K+</span>
        <span class="stat-label">Panels Analysed</span>
      </div>
      <div class="stat-item">
        <span class="stat-value"><span id="c3">0</span>%</span>
        <span class="stat-label">Energy Recovered</span>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     ABOUT
     ============================================================ -->
<section id="about" class="section">
  <div class="container">
    <div class="about-grid">
      <div class="about-img-wrap reveal">
        <img src="https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=720&q=70" alt="Solar farm">
        <div class="about-img-badge">
          <div class="val">+30%</div>
          <div class="lbl">Efficiency Gain</div>
        </div>
      </div>
      <div class="reveal">
        <span class="section-label">About SolarScan AI</span>
        <h2 class="section-heading">Intelligent Dust Detection for Maximum Solar Output</h2>
        <p class="section-sub">
          SolarScan AI uses MobileNetV2 deep learning with Grad-CAM heatmaps to pinpoint dust on solar panels from a
          single video. Frame by frame, our system predicts efficiency loss, estimates power reduction,
          and generates a tailored cleaning schedule.
        </p>
        <ul class="about-list">
          <li><i class="fas fa-check-circle"></i> Upload any MP4, AVI, MOV, or MKV panel video</li>
          <li><i class="fas fa-check-circle"></i> Deep learning model localises dust with pixel-level heatmaps</li>
          <li><i class="fas fa-check-circle"></i> Receives an instant efficiency score and maintenance timeline</li>
          <li><i class="fas fa-check-circle"></i> Compatible with rooftop, ground-mount, and drone imagery</li>
          <li><i class="fas fa-check-circle"></i> Full prediction history logged server-side for audits</li>
        </ul>
        <a href="#features" class="btn btn-primary" style="margin-top:28px"><i class="fas fa-arrow-right"></i> Explore Features</a>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     FEATURES
     ============================================================ -->
<section id="features" class="section" style="background:var(--bg2)">
  <div class="container">
    <div class="center reveal">
      <span class="section-label">Features</span>
      <h2 class="section-heading">Everything You Need to Monitor Solar Health</h2>
      <p class="section-sub">Six intelligent tools working together to keep your panels operating at peak performance.</p>
    </div>
    <div class="features-grid">
      <div class="glass feat-card reveal">
        <div class="feat-icon icon-green"><i class="fas fa-microchip"></i></div>
        <h3>AI Dust Detection</h3>
        <p>MobileNetV2 fine-tuned on thousands of real-world panel images classifies dust with up to 97% precision in under 2 seconds.</p>
      </div>
      <div class="glass feat-card reveal">
        <div class="feat-icon icon-blue"><i class="fas fa-film"></i></div>
        <h3>Video Upload</h3>
        <p>Drag-and-drop MP4, AVI, MOV, or MKV video files. Supports footage from smartphones, cameras, and aerial drones alike.</p>
      </div>
      <div class="glass feat-card reveal">
        <div class="feat-icon icon-gold"><i class="fas fa-percentage"></i></div>
        <h3>Dust Percentage Estimation</h3>
        <p>Quantifies the proportion of the panel surface obscured by particulates using Grad-CAM activation maps.</p>
      </div>
      <div class="glass feat-card reveal">
        <div class="feat-icon icon-red"><i class="fas fa-bolt"></i></div>
        <h3>Power Loss Prediction</h3>
        <p>Translates dust levels into estimated kilowatt-hour losses per day, giving you a clear financial case for maintenance.</p>
      </div>
      <div class="glass feat-card reveal">
        <div class="feat-icon icon-orange"><i class="fas fa-broom"></i></div>
        <h3>Cleaning Recommendation</h3>
        <p>Generates a prioritised maintenance schedule — from "no action needed" to "clean within 24 hours" — based on detected severity.</p>
      </div>
      <div class="glass feat-card reveal">
        <div class="feat-icon icon-purple"><i class="fas fa-chart-line"></i></div>
        <h3>Analytics Dashboard</h3>
        <p>Track efficiency trends, dust ratios, and maintenance history across multiple panels in a unified, real-time view.</p>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     HOW IT WORKS
     ============================================================ -->
<section id="how" class="section">
  <div class="container">
    <div class="center reveal">
      <span class="section-label">Process</span>
      <h2 class="section-heading">From Upload to Insight in Five Steps</h2>
      <p class="section-sub">A streamlined pipeline that turns a video into actionable maintenance intelligence, frame by frame.</p>
    </div>
    <div class="timeline reveal">
      <div class="tl-step">
        <div class="tl-circle"><i class="fas fa-video"></i><span class="tl-num">1</span></div>
        <h4>Upload Video</h4>
        <p>Drag & drop your panel photo or select from device storage</p>
      </div>
      <div class="tl-connector"></div>
      <div class="tl-step">
        <div class="tl-circle"><i class="fas fa-brain"></i><span class="tl-num">2</span></div>
        <h4>AI Analysis</h4>
        <p>MobileNetV2 processes each video frame through 150+ convolutional layers</p>
      </div>
      <div class="tl-connector"></div>
      <div class="tl-step">
        <div class="tl-circle"><i class="fas fa-search"></i><span class="tl-num">3</span></div>
        <h4>Dust Detection</h4>
        <p>Grad-CAM heatmap highlights contaminated regions in real time</p>
      </div>
      <div class="tl-connector"></div>
      <div class="tl-step">
        <div class="tl-circle"><i class="fas fa-chart-pie"></i><span class="tl-num">4</span></div>
        <h4>Efficiency Score</h4>
        <p>Power-loss percentage and health rating generated automatically</p>
      </div>
      <div class="tl-connector"></div>
      <div class="tl-step">
        <div class="tl-circle"><i class="fas fa-clipboard-check"></i><span class="tl-num">5</span></div>
        <h4>Recommendation</h4>
        <p>Tailored cleaning schedule delivered with confidence interval</p>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     STATS
     ============================================================ -->
<section id="stats" class="section" style="background:var(--bg2)">
  <div class="container">
    <div class="center reveal">
      <span class="section-label">Why It Matters</span>
      <h2 class="section-heading">The Real Cost of Dusty Solar Panels</h2>
      <p class="section-sub">Dust is the silent thief of solar energy. The numbers make the case for proactive monitoring.</p>
    </div>
    <div class="stats-grid">
      <div class="glass stat-card reveal">
        <span class="s-icon">⚡</span>
        <div class="s-val" data-target="30">0%</div>
        <div class="s-label">Max Energy Loss from Dust</div>
      </div>
      <div class="glass stat-card reveal">
        <span class="s-icon">📉</span>
        <div class="s-val" data-target="25">0%</div>
        <div class="s-label">Avg Efficiency Reduction</div>
      </div>
      <div class="glass stat-card reveal">
        <span class="s-icon">💰</span>
        <div class="s-val" data-target="40">0%</div>
        <div class="s-label">Lower Maintenance Cost</div>
      </div>
      <div class="glass stat-card reveal">
        <span class="s-icon">🌍</span>
        <div class="s-val" data-target="97">0%</div>
        <div class="s-label">Model Detection Accuracy</div>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     DASHBOARD PREVIEW
     ============================================================ -->
<section id="dashboard" class="section">
  <div class="container">
    <div class="center reveal">
      <span class="section-label">Dashboard</span>
      <h2 class="section-heading">Real-Time Panel Intelligence at a Glance</h2>
      <p class="section-sub">An interactive control centre showing dust levels, efficiency scores, and maintenance status.</p>
    </div>
    <div class="dash-wrap reveal">
      <div class="dash-topbar">
        <div class="dash-dot d1"></div>
        <div class="dash-dot d2"></div>
        <div class="dash-dot d3"></div>
        <span class="dash-title">SolarScan AI — Panel Analysis Dashboard</span>
      </div>
      <div class="dash-body">
        <div class="dash-left">
          <div class="dash-panel-img">
            <div class="dust-overlay"></div>
            <div class="scan-line"></div>
          </div>
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
            <div>
              <div style="font-size:0.8rem;color:var(--muted)">Panel ID</div>
              <div style="font-size:0.95rem;font-weight:600">SP-0047-C</div>
            </div>
            <span class="label-chip chip-warn">⚠ Moderate Dust</span>
          </div>
          <div class="progress-bar-wrap">
            <div class="pb-label"><span>Dust Level</span><span style="color:var(--warn)">38%</span></div>
            <div class="pb-track"><div class="pb-fill fill-warn" id="pb-dust" style="width:0%"></div></div>
          </div>
          <div class="progress-bar-wrap">
            <div class="pb-label"><span>Panel Efficiency</span><span style="color:var(--primary)">74%</span></div>
            <div class="pb-track"><div class="pb-fill fill-green" id="pb-eff" style="width:0%"></div></div>
          </div>
          <div class="progress-bar-wrap">
            <div class="pb-label"><span>AI Confidence</span><span style="color:var(--secondary)">91%</span></div>
            <div class="pb-track"><div class="pb-fill" style="background:linear-gradient(90deg,var(--secondary),#60aaff);width:0%" id="pb-conf"></div></div>
          </div>
        </div>
        <div class="dash-right">
          <div class="donut-wrap">
            <div class="donut">
              <svg width="120" height="120" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="48" fill="none" stroke="rgba(255,255,255,0.07)" stroke-width="12"/>
                <circle cx="60" cy="60" r="48" fill="none" stroke="var(--warn)" stroke-width="12"
                  stroke-dasharray="301.6" stroke-dashoffset="301.6" id="donut-arc"
                  stroke-linecap="round" style="transition:stroke-dashoffset 1.4s ease"/>
              </svg>
              <div class="donut-text">
                <span class="d-val" id="donut-val">0%</span>
                <span class="d-lbl">Dust Level</span>
              </div>
            </div>
          </div>
          <div class="dash-metric-row">
            <span class="dash-metric-label"><i class="fas fa-bolt" style="color:var(--danger);margin-right:6px"></i>Estimated Power Loss</span>
            <span class="dash-metric-val" style="color:var(--danger)">−1.8 kWh/day</span>
          </div>
          <div class="dash-metric-row">
            <span class="dash-metric-label"><i class="fas fa-calendar-alt" style="color:var(--primary);margin-right:6px"></i>Next Maintenance</span>
            <span class="dash-metric-val" style="color:var(--primary)">In 3 Days</span>
          </div>
          <div class="dash-metric-row">
            <span class="dash-metric-label"><i class="fas fa-heart" style="color:var(--warn);margin-right:6px"></i>Health Score</span>
            <span class="dash-metric-val" style="color:var(--warn)">74 / 100</span>
          </div>
          <div class="dash-metric-row">
            <span class="dash-metric-label"><i class="fas fa-check-circle" style="color:var(--muted);margin-right:6px"></i>Cleaning Status</span>
            <span class="label-chip chip-warn" style="font-size:0.72rem">Monitor</span>
          </div>
          <div style="height:1px;background:var(--border);margin:16px 0"></div>
          <div style="font-size:0.78rem;color:var(--muted);margin-bottom:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em">Recent Predictions</div>
          <div style="display:flex;flex-direction:column;gap:8px">
            <div style="display:flex;justify-content:space-between;font-size:0.78rem"><span style="color:var(--muted)">Panel SP-0043</span><span class="label-chip chip-ok" style="font-size:0.65rem;padding:2px 8px">Clean</span></div>
            <div style="display:flex;justify-content:space-between;font-size:0.78rem"><span style="color:var(--muted)">Panel SP-0046</span><span class="label-chip chip-warn" style="font-size:0.65rem;padding:2px 8px">Monitor</span></div>
            <div style="display:flex;justify-content:space-between;font-size:0.78rem"><span style="color:var(--muted)">Panel SP-0031</span><span class="label-chip chip-bad" style="font-size:0.65rem;padding:2px 8px">Clean Now</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     SAMPLE RESULTS
     ============================================================ -->
<section id="results" class="section" style="background:var(--bg2)">
  <div class="container">
    <div class="center reveal">
      <span class="section-label">Sample Results</span>
      <h2 class="section-heading">See What SolarScan AI Detects</h2>
      <p class="section-sub">Three representative outcomes, from spotless to critically soiled panels.</p>
    </div>
    <div class="results-grid">
      <div class="glass res-card reveal">
        <span class="res-icon">☀️</span>
        <div class="res-title">Clean Panel</div>
        <div class="res-dust">Dust level: 5%</div>
        <div class="res-bar-wrap">
          <div class="res-bar" style="width:5%;background:linear-gradient(90deg,var(--primary),#00e090)"></div>
        </div>
        <span class="res-status" style="background:rgba(0,208,132,0.12);color:var(--primary)">✓ Excellent — No Action</span>
        <div style="font-size:0.78rem;color:var(--muted);margin-top:14px">Efficiency: 98% · AI Confidence: 96%</div>
      </div>
      <div class="glass res-card reveal">
        <span class="res-icon">🌤️</span>
        <div class="res-title">Moderate Dust</div>
        <div class="res-dust">Dust level: 38%</div>
        <div class="res-bar-wrap">
          <div class="res-bar" style="width:38%;background:linear-gradient(90deg,var(--warn),#ffd060)"></div>
        </div>
        <span class="res-status" style="background:rgba(255,176,32,0.12);color:var(--warn)">⚠ Monitor — Clean in 3 Days</span>
        <div style="font-size:0.78rem;color:var(--muted);margin-top:14px">Efficiency: 74% · AI Confidence: 91%</div>
      </div>
      <div class="glass res-card reveal">
        <span class="res-icon">🌫️</span>
        <div class="res-title">Heavy Dust</div>
        <div class="res-dust">Dust level: 72%</div>
        <div class="res-bar-wrap">
          <div class="res-bar" style="width:72%;background:linear-gradient(90deg,var(--danger),#ff7a8a)"></div>
        </div>
        <span class="res-status" style="background:rgba(255,77,106,0.12);color:var(--danger)">✗ Critical — Clean Immediately</span>
        <div style="font-size:0.78rem;color:var(--muted);margin-top:14px">Efficiency: 41% · AI Confidence: 99%</div>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     BENEFITS
     ============================================================ -->
<section id="benefits" class="section">
  <div class="container">
    <div class="center reveal">
      <span class="section-label">Benefits</span>
      <h2 class="section-heading">Why Teams Choose SolarScan AI</h2>
      <p class="section-sub">From utility-scale farms to residential rooftops, our platform delivers measurable results.</p>
    </div>
    <div class="benefits-grid">
      <div class="glass ben-card reveal">
        <i class="fas fa-sun ben-icon"></i>
        <h3>Increase Solar Efficiency</h3>
        <p>Timely detection and cleaning can recover up to 30% of lost output, translating directly into higher energy yield and revenue from your installation.</p>
      </div>
      <div class="glass ben-card reveal">
        <i class="fas fa-piggy-bank ben-icon"></i>
        <h3>Reduce Cleaning Costs</h3>
        <p>AI-guided scheduling eliminates unnecessary maintenance visits. Clean only when the data says so, cutting operational expenses by up to 40%.</p>
      </div>
      <div class="glass ben-card reveal">
        <i class="fas fa-leaf ben-icon"></i>
        <h3>Improve Renewable Output</h3>
        <p>Maximise the clean energy contribution of every panel in your fleet, strengthening your sustainability KPIs and reducing carbon footprint.</p>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     FAQ
     ============================================================ -->
<section id="faq" class="section" style="background:var(--bg2)">
  <div class="container">
    <div class="center reveal">
      <span class="section-label">FAQ</span>
      <h2 class="section-heading">Common Questions</h2>
      <p class="section-sub">Everything you need to know before getting started.</p>
    </div>
    <div class="faq-list">
      <div class="faq-item reveal">
        <button class="faq-q">How does AI detect dust on solar panels?<i class="fas fa-chevron-down faq-icon"></i></button>
        <div class="faq-a">Our MobileNetV2 model was trained on over 50,000 labelled images of clean and dusty solar panels under varying lighting conditions. It learns to recognise texture and colour patterns associated with particulate contamination, then uses Grad-CAM to produce a spatial heatmap highlighting the affected zones.</div>
      </div>
      <div class="faq-item reveal">
        <button class="faq-q">What image formats are supported?<i class="fas fa-chevron-down faq-icon"></i></button>
        <div class="faq-a">The image upload endpoint accepts JPG, JPEG, PNG, BMP, and WEBP files. Video analysis supports MP4, AVI, MOV, and MKV formats. File size is capped at 50 MB per upload; larger drone surveys should be broken into individual panel frames before submission.</div>
      </div>
      <div class="faq-item reveal">
        <button class="faq-q">How accurate is the detection model?<i class="fas fa-chevron-down faq-icon"></i></button>
        <div class="faq-a">On our internal benchmark dataset the model achieves 97.2% top-1 accuracy, 96.8% precision, and an F1 score of 0.971. Performance can vary with very low-resolution images or extreme lighting; we recommend images of at least 640 × 480 pixels captured in daylight for optimal results.</div>
      </div>
      <div class="faq-item reveal">
        <button class="faq-q">Can drone images be used?<i class="fas fa-chevron-down faq-icon"></i></button>
        <div class="faq-a">Yes. SolarScan AI was specifically validated with nadir (top-down) drone imagery from DJI Phantom 4 and Mavic 3 platforms. For best results, fly at 5–15 m altitude with the gimbal angled directly downward. The video analysis feature is particularly suited to automated drone survey footage.</div>
      </div>
      <div class="faq-item reveal">
        <button class="faq-q">Does it work on rooftop panels?<i class="fas fa-chevron-down faq-icon"></i></button>
        <div class="faq-a">Absolutely. Ground-mount, rooftop, and façade-integrated panels are all supported. The model handles panels at various tilt angles (0°–45°) and in both portrait and landscape orientation. For rooftop installations, a standard smartphone photo taken from a safe vantage point at close range delivers excellent results.</div>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     CTA
     ============================================================ -->
<section id="cta" class="section">
  <div class="cta-bg"></div>
  <div class="cta-glow"></div>
  <div class="container" style="position:relative;z-index:1">
    <div class="reveal">
      <span class="section-label">Get Started Today</span>
      <h2 class="section-heading">Start Detecting Solar Panel Dust Today</h2>
      <p class="section-sub" style="margin:0 auto 36px;max-width:480px">Upload your first panel video in under a minute and receive a full AI diagnostic report — completely free.</p>
      <div style="display:flex;gap:14px;justify-content:center;flex-wrap:wrap">
        <a href="#" class="btn btn-primary" id="uploadVideoBtn" style="font-size:1rem;padding:15px 36px"><i class="fas fa-video"></i> Upload Your Video</a>
        <a href="#features" class="btn btn-outline" style="font-size:1rem;padding:15px 36px"><i class="fas fa-book-open"></i> Learn More</a>
      </div>
      
      <!-- Result Container -->
      <div id="resultContainer" style="display:none; margin-top:40px; padding:25px; background:rgba(0,208,132,0.08); border:1px solid rgba(0,208,132,0.2); border-radius:20px; backdrop-filter:blur(10px);">
        <div id="resultContent"></div>
      </div>
    </div>
  </div>
</section>

<!-- ============================================================
     FOOTER
     ============================================================ -->
<footer>
  <div class="container">
    <div class="footer-grid">
      <div class="footer-brand">
        <a href="#" class="logo" style="margin-bottom:12px;display:inline-flex">
          <div class="logo-icon"><i class="fas fa-sun"></i></div>
          Solar<span>Scan</span> AI
        </a>
        <p>AI-powered solar panel dust detection system. Built with MobileNetV2, Grad-CAM, and a passion for renewable energy.</p>
        <div class="footer-social">
          <a href="#"><i class="fab fa-twitter"></i></a>
          <a href="#"><i class="fab fa-linkedin-in"></i></a>
          <a href="#"><i class="fab fa-github"></i></a>
          <a href="#"><i class="fab fa-youtube"></i></a>
        </div>
      </div>
      <div class="footer-col">
        <h5>Quick Links</h5>
        <ul>
          <li><a href="#about">About</a></li>
          <li><a href="#features">Features</a></li>
          <li><a href="#how">How It Works</a></li>
          <li><a href="#dashboard">Dashboard</a></li>
          <li><a href="#faq">FAQ</a></li>
        </ul>
      </div>
      <div class="footer-col">
        <h5>Product</h5>
        <ul>
          <li><a href="#">Image Analysis</a></li>
          <li><a href="#">Video Analysis</a></li>
          <li><a href="#">API Access</a></li>
          <li><a href="#">History & Reports</a></li>
          <li><a href="#">Pricing</a></li>
        </ul>
      </div>
      <div class="footer-col">
        <h5>Contact</h5>
        <ul>
          <li><a href="mailto:hello@solarscan.ai"><i class="fas fa-envelope" style="margin-right:6px"></i>hello@solarscan.ai</a></li>
          <li><a href="#"><i class="fas fa-map-marker-alt" style="margin-right:6px"></i>Bangalore, India</a></li>
          <li><a href="#"><i class="fas fa-headset" style="margin-right:6px"></i>Support Centre</a></li>
          <li><a href="#">Privacy Policy</a></li>
          <li><a href="#">Terms of Service</a></li>
        </ul>
      </div>
    </div>
    <div class="footer-bottom">
      <span>© 2026 SolarScan AI. All rights reserved.</span>
      <div style="display:flex;gap:20px">
        <a href="#">Privacy Policy</a>
        <a href="#">Terms</a>
        <a href="#">Cookie Settings</a>
      </div>
    </div>
  </div>
</footer>

<!-- ============================================================
     JAVASCRIPT
     ============================================================ -->
<script>
/* --- PARTICLE CANVAS --- */
(function() {
  const canvas = document.getElementById('particle-canvas');
  const ctx = canvas.getContext('2d');
  let particles = [];
  const W = () => canvas.width = window.innerWidth;
  const H = () => canvas.height = window.innerHeight;
  W(); H();
  window.addEventListener('resize', () => { W(); H(); });

  function Particle() {
    this.x = Math.random() * canvas.width;
    this.y = Math.random() * canvas.height;
    this.r = Math.random() * 1.5 + 0.3;
    this.vx = (Math.random() - 0.5) * 0.18;
    this.vy = (Math.random() - 0.5) * 0.18;
    this.alpha = Math.random() * 0.55 + 0.1;
    this.color = Math.random() > 0.6 ? '#00D084' : '#1E90FF';
  }
  for (let i = 0; i < 120; i++) particles.push(new Particle());

  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.globalAlpha = p.alpha;
      ctx.fill();
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;
    });
    ctx.globalAlpha = 1;
    requestAnimationFrame(animate);
  }
  animate();
})();

/* --- NAVBAR SCROLL --- */
window.addEventListener('scroll', () => {
  const navbar = document.getElementById('navbar');
  if (navbar) navbar.classList.toggle('scrolled', window.scrollY > 40);
});

/* --- MOBILE MENU --- */
function closeMobile() {
  const mobileMenu = document.getElementById('mobileMenu');
  if (mobileMenu) mobileMenu.classList.remove('open');
}

const hamburger = document.getElementById('hamburger');
const mobileClose = document.getElementById('mobileClose');

if (hamburger) {
  hamburger.addEventListener('click', () => {
    const mobileMenu = document.getElementById('mobileMenu');
    if (mobileMenu) mobileMenu.classList.add('open');
  });
}
if (mobileClose) {
  mobileClose.addEventListener('click', closeMobile);
}

/* --- REVEAL ON SCROLL --- */
const revealEls = document.querySelectorAll('.reveal');
const observer = new IntersectionObserver((entries) => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) {
      setTimeout(() => e.target.classList.add('visible'), i * 60);
      observer.unobserve(e.target);
    }
  });
}, { threshold: 0.12 });
revealEls.forEach(el => observer.observe(el));

/* --- ANIMATED COUNTERS (hero) --- */
function animCount(id, target, suffix, duration) {
  const el = document.getElementById(id);
  if (!el) return;
  let start = 0, step = target / (duration / 16);
  const t = setInterval(() => {
    start = Math.min(start + step, target);
    el.textContent = Math.floor(start) + suffix;
    if (start >= target) clearInterval(t);
  }, 16);
}

const heroObserver = new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting) {
    animCount('c1', 97, '%', 1400);
    animCount('c2', 250, 'K+', 1600);
    animCount('c3', 30, '%', 1200);
    heroObserver.disconnect();
  }
}, { threshold: 0.5 });
const heroElement = document.getElementById('hero');
if (heroElement) heroObserver.observe(heroElement);

/* --- ANIMATED COUNTERS (stats section) --- */
const statCards = document.querySelectorAll('.stat-card[data-target]');
const statObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const el = entry.target;
      const target = parseInt(el.dataset.target);
      const valEl = el.querySelector('.s-val');
      if (!valEl) return;
      let cur = 0;
      const step = target / 60;
      const t = setInterval(() => {
        cur = Math.min(cur + step, target);
        valEl.textContent = Math.floor(cur) + '%';
        if (cur >= target) clearInterval(t);
      }, 18);
      statObserver.unobserve(el);
    }
  });
}, { threshold: 0.4 });
statCards.forEach(c => statObserver.observe(c));

/* --- DASHBOARD ANIMATIONS --- */
const dashObserver = new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting) {
    setTimeout(() => {
      const pbDust = document.getElementById('pb-dust');
      const pbEff = document.getElementById('pb-eff');
      const pbConf = document.getElementById('pb-conf');
      if (pbDust) pbDust.style.width = '38%';
      if (pbEff) pbEff.style.width = '74%';
      if (pbConf) pbConf.style.width = '91%';

      const donutArc = document.getElementById('donut-arc');
      if (donutArc) {
        const circ = 2 * Math.PI * 48;
        const pct = 0.38;
        donutArc.style.strokeDashoffset = circ * (1 - pct);
      }

      let v = 0;
      const donutVal = document.getElementById('donut-val');
      const t = setInterval(() => {
        v = Math.min(v + 1, 38);
        if (donutVal) donutVal.textContent = v + '%';
        if (v >= 38) clearInterval(t);
      }, 28);
    }, 300);
    dashObserver.disconnect();
  }
}, { threshold: 0.3 });
const dashboardElement = document.getElementById('dashboard');
if (dashboardElement) dashObserver.observe(dashboardElement);

/* --- FAQ ACCORDION --- */
document.querySelectorAll('.faq-q').forEach(btn => {
  btn.addEventListener('click', () => {
    const item = btn.parentElement;
    const open = item.classList.contains('open');
    document.querySelectorAll('.faq-item.open').forEach(i => i.classList.remove('open'));
    if (!open) item.classList.add('open');
  });
});

/* --- AJAX UPLOAD LOGIC WITH RESULT DISPLAY --- */
const ctaSection = document.getElementById('cta');
if (ctaSection && !document.getElementById('videoInput')) {
  ctaSection.insertAdjacentHTML('beforeend', '<input type="file" id="videoInput" accept="video/*" style="display:none">');
}

const uploadBtn = document.getElementById('uploadVideoBtn');
const resultContainer = document.getElementById('resultContainer');
const resultContent = document.getElementById('resultContent');

if (uploadBtn) {
  uploadBtn.addEventListener('click', (e) => {
    e.preventDefault();
    const videoInput = document.getElementById('videoInput');
    if (videoInput) videoInput.click();
  });
}

const videoInput = document.getElementById('videoInput');
if (videoInput) {
  videoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Show loading state
    if (resultContainer && resultContent) {
      resultContainer.style.display = 'block';
      resultContent.innerHTML = `
        <div style="text-align:center">
          <div class="loader"></div>
          <p style="margin-top:15px; color:var(--muted)">🔍 Analyzing video "${file.name}"... Please wait.</p>
          <p style="font-size:0.8rem; color:var(--muted); margin-top:10px">Processing frames with AI model...</p>
        </div>
      `;
      resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/predict/video', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      
      if (response.ok) {
        console.log("Success:", data);
        
        const summary = data.summary || {};
        const isDusty = data.label === "Dusty";
        const dustPercent = summary.dust_ratio || 0;
        const framesAnalyzed = summary.frames_analyzed || 0;
        const efficiency = summary.efficiency || (isDusty ? 65 : 92);
        const maintenance = summary.maintenance || (isDusty ? "Cleaning Recommended" : "No Cleaning Required");
        
        // Display beautiful results
        resultContent.innerHTML = `
          <div style="text-align:center">
            <div class="result-badge">
              ${isDusty ? '⚠️ 🌫️' : '✅ ☀️'}
            </div>
            <div class="result-title" style="color:${isDusty ? 'var(--danger)' : 'var(--primary)'}">
              ${data.label.toUpperCase()}
            </div>
            <p style="color:var(--muted); margin-bottom:20px">
              ${isDusty ? 'Dust detected on solar panel surface' : 'Panel is clean and operating efficiently'}
            </p>
            <div class="result-grid">
              <div class="result-card">
                <div class="result-card-label">Dust Ratio</div>
                <div class="result-card-value">${dustPercent}%</div>
              </div>
              <div class="result-card">
                <div class="result-card-label">Frames Analyzed</div>
                <div class="result-card-value">${framesAnalyzed}</div>
              </div>
              <div class="result-card">
                <div class="result-card-label">Efficiency</div>
                <div class="result-card-value">${efficiency}%</div>
              </div>
              <div class="result-card">
                <div class="result-card-label">Maintenance</div>
                <div class="result-card-value" style="font-size:0.85rem">${maintenance}</div>
              </div>
            </div>
            ${data.heatmap ? `<div style="margin-top:20px">
              <img src="data:image/png;base64,${data.heatmap}" alt="Heatmap" style="max-width:100%; border-radius:12px; margin-top:15px">
            </div>` : ''}
            <button onclick="document.getElementById('resultContainer').style.display='none'" 
                    style="margin-top:20px; background:var(--card); border:1px solid var(--border); 
                           color:var(--text); padding:8px 20px; border-radius:25px; cursor:pointer">
              Close
            </button>
          </div>
        `;
      } else {
        resultContent.innerHTML = `
          <div style="text-align:center">
            <div class="result-badge">❌</div>
            <div class="result-title" style="color:var(--danger)">Error</div>
            <p style="color:var(--muted)">${data.error || 'Something went wrong'}</p>
            <button onclick="document.getElementById('resultContainer').style.display='none'" 
                    style="margin-top:20px; background:var(--card); border:1px solid var(--border); 
                           color:var(--text); padding:8px 20px; border-radius:25px; cursor:pointer">
              Close
            </button>
          </div>
        `;
      }
    } catch (err) {
      console.error(err);
      resultContent.innerHTML = `
        <div style="text-align:center">
          <div class="result-badge">🔌</div>
          <div class="result-title" style="color:var(--warn)">Connection Failed</div>
          <p style="color:var(--muted)">Failed to connect to the backend. Make sure the server is running.</p>
          <p style="font-size:0.8rem; color:var(--muted)">Run: python app.py</p>
          <button onclick="document.getElementById('resultContainer').style.display='none'" 
                  style="margin-top:20px; background:var(--card); border:1px solid var(--border); 
                         color:var(--text); padding:8px 20px; border-radius:25px; cursor:pointer">
            Close
          </button>
        </div>
      `;
    }
  });
}
</script>
</body>
</html>'''