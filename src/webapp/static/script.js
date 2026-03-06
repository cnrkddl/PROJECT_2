document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('mainVideo');
    const overlayContainer = document.getElementById('overlayContainer');
    const timelineList = document.getElementById('timelineList');

    // Attempt to load timetable data
    let timetable = [];
    try {
        const dataStr = document.getElementById('timetableData').textContent;
        timetable = JSON.parse(dataStr);
    } catch (e) {
        console.error("Failed to parse timetable JSON", e);
    }

    // Track active banners to avoid recreating DOM elements on every frame
    const activeBanners = new Set();
    const bannerElements = new Map();

    // Mapping CSV UI type to CSS classes
    function getCssClassForUiType(uiType) {
        if (uiType.includes('하단 팝업') || uiType.includes('bottom')) {
            return 'ui-pop-up-bottom';
        }
        if (uiType.includes('우측 상단') || uiType.includes('L자') || uiType.includes('top-right')) {
            return 'ui-l-banner-top-right';
        }
        // default fallback
        return 'ui-pop-up-bottom';
    }

    // Format seconds to MM:SS
    function formatTime(seconds) {
        const m = Math.floor(seconds / 60).toString().padStart(2, '0');
        const s = Math.floor(seconds % 60).toString().padStart(2, '0');
        return `${m}:${s}`;
    }

    // Initialize Sidebar Timeline
    function initSidebar() {
        if (timetable.length === 0) {
            timelineList.innerHTML = '<li>No ads scheduled.</li>';
            return;
        }

        timetable.forEach((ad, index) => {
            const li = document.createElement('li');
            li.className = 'timeline-item';
            li.id = `timeline-item-${index}`;

            li.innerHTML = `
                <span class="time-badge">${formatTime(ad.start_time)} - ${formatTime(ad.end_time)}</span>
                ${ad.image_url ? `<img class="timeline-thumb" src="${ad.image_url}" alt="${ad.item_name}">` : ''}
                <span class="item-name">${ad.item_name}</span>
                <span class="ui-type">${ad.ui_type}</span>
                ${ad.reason ? `<p class="ad-reason">${ad.reason}</p>` : ''}
            `;

            // Stagger animation delay on load
            li.style.animation = `fadeInDown 0.5s ease-out ${index * 0.15}s both`;

            // Allow clicking timeline item to jump video
            li.addEventListener('click', () => {
                video.currentTime = ad.start_time;
                video.play();
            });

            timelineList.appendChild(li);

            // Pre-create the DOM elements for overlays (but hidden)
            const banner = document.createElement('div');
            banner.className = `ad-banner ${getCssClassForUiType(ad.ui_type)}`;
            banner.id = `ad-banner-${index}`;
            banner.innerHTML = `
                <div class="ad-label">AD</div>
                <img src="${ad.image_url}" alt="${ad.item_name}">
            `;

            overlayContainer.appendChild(banner);
            bannerElements.set(index, banner);
        });
    }

    initSidebar();

    // Update loop based on video time
    video.addEventListener('timeupdate', () => {
        const currentTime = video.currentTime;

        timetable.forEach((ad, index) => {
            const isActiveTime = currentTime >= ad.start_time && currentTime <= ad.end_time;
            const isCurrentlyActive = activeBanners.has(index);

            // Need to show
            if (isActiveTime && !isCurrentlyActive) {
                activeBanners.add(index);
                const bannerEl = bannerElements.get(index);
                if (bannerEl) bannerEl.classList.add('active');

                // Highlight sidebar list item
                const li = document.getElementById(`timeline-item-${index}`);
                if (li) {
                    li.classList.add('active-item');
                    li.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }
            // Need to hide
            else if (!isActiveTime && isCurrentlyActive) {
                activeBanners.delete(index);
                const bannerEl = bannerElements.get(index);
                if (bannerEl) bannerEl.classList.remove('active');

                // Remove highlight from sidebar list item
                const li = document.getElementById(`timeline-item-${index}`);
                if (li) li.classList.remove('active-item');
            }
        });
    });
});
