// AntiTerror Dashboard JS

// Format Unix timestamp to human-readable
function formatTs(ts) {
    if (!ts) return '-';
    const d = new Date(ts * 1000);
    return d.toLocaleString('ru-RU', {
        day: '2-digit', month: '2-digit', year: 'numeric',
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
}

// Convert all .timestamp elements on page load
document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.timestamp').forEach(el => {
        const ts = parseFloat(el.dataset.ts);
        if (!isNaN(ts)) {
            el.textContent = formatTs(ts);
        }
    });

    // Update clock
    const clock = document.getElementById('clock');
    if (clock) {
        setInterval(() => {
            clock.textContent = new Date().toLocaleString('ru-RU', {
                hour: '2-digit', minute: '2-digit', second: '2-digit'
            });
        }, 1000);
    }

    // Highlight active nav link
    const path = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        const href = link.getAttribute('href');
        if (href === path || (href !== '/' && path.startsWith(href))) {
            link.classList.add('active');
        }
    });
});
