(() => {
  const rally = document.querySelector('#rally');
  const seek = document.querySelector('#seek');
  const elapsed = document.querySelector('#elapsed');
  const play = document.querySelector('#play');
  const modeLabel = document.querySelector('#clip-mode');
  const buttons = [...document.querySelectorAll('[data-mode]')];
  const views = {
    detection: {
      video: 'assets/rally.mp4', poster: 'assets/annotated.webp',
      label: 'YOLO26 overlay', description: 'Paris rally with YOLO26 detection boxes'
    },
    original: {
      video: 'assets/original.mp4', poster: 'assets/original.webp',
      label: 'Original broadcast', description: 'The same Paris rally without detection boxes'
    }
  };

  let position = 4;
  let resume = false;
  let loading = true;
  let mode = 'detection';
  const stamp = time => '0:' + String(Math.floor(time)).padStart(2, '0');
  function updateTime(time) {
    seek.value = time;
    elapsed.textContent = stamp(time);
    seek.setAttribute('aria-valuetext', time.toFixed(1) + ' seconds of 10 seconds');
  }
  function updateButton() {
    const running = !rally.paused && !rally.ended;
    const label = running ? 'Pause rally' : 'Play rally';
    play.firstElementChild.className = running ? 'pause-icon' : 'play-icon';
    play.setAttribute('aria-label', label);
    play.title = label;
  }
  function start() {
    rally.play().catch(() => {
      resume = false;
      updateButton();
    });
  }
  function ready() {
    if (!loading || !Number.isFinite(rally.duration)) return;
    loading = false;
    rally.currentTime = Math.min(position, rally.duration);
    updateTime(rally.currentTime);
    if (resume) start();
  }
  rally.addEventListener('loadeddata', ready);
  rally.addEventListener('timeupdate', () => {
    if (loading) return;
    position = rally.currentTime;
    updateTime(position);
  });
  for (const event of ['play', 'pause', 'ended']) rally.addEventListener(event, updateButton);
  play.addEventListener('click', () => {
    if (!rally.paused) {
      resume = false;
      rally.pause();
      return;
    }
    resume = true;
    if (rally.ended) {
      position = 0;
      rally.currentTime = 0;
    }
    start();
  });
  seek.addEventListener('input', () => {
    position = Number(seek.value);
    updateTime(position);
    if (!loading) rally.currentTime = position;
  });
  for (const button of buttons) {
    button.addEventListener('click', () => {
      const next = button.dataset.mode;
      if (next === mode) return;
      if (!loading) {
        position = rally.currentTime;
        resume = !rally.paused && !rally.ended;
      }
      loading = true;
      rally.pause();
      mode = next;
      const view = views[mode];
      rally.poster = view.poster;
      rally.src = view.video;
      rally.setAttribute('aria-label', view.description);
      modeLabel.textContent = view.label;
      for (const choice of buttons) {
        choice.setAttribute('aria-pressed', String(choice === button));
      }
      rally.load();
    });
  }
  rally.addEventListener('error', () => {
    // Preserve a direct media link when the browser lacks an H.264 decoder.
    const view = views[mode];
    modeLabel.replaceChildren();
    const link = document.createElement('a');
    link.href = view.video;
    link.textContent = 'Open rally video (MP4)';
    modeLabel.append(link);
  });
  rally.controls = false;
  document.documentElement.classList.add('js-ready');
  if (rally.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) ready();
  updateTime(position);
  updateButton();

  const navigation = [...document.querySelectorAll('.head-inner nav a')];
  const sections = navigation.map(link => document.querySelector(link.hash));
  function updateNavigation() {
    const visible = sections.filter(section => section.getBoundingClientRect().top < window.innerHeight * .35);
    const current = visible.at(-1) || sections[0];
    for (const link of navigation) {
      const active = link.hash === '#' + current.id;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }
  const observer = new IntersectionObserver(updateNavigation, {rootMargin: '0px 0px -65% 0px'});
  sections.forEach(section => observer.observe(section));
  updateNavigation();
})();
