// Configuration
const circleRadius = Math.min(window.innerWidth, window.innerHeight) * 0.4;
const arrowSize = circleRadius * 0.05;

// Tracking variables
let currentLineIndex = -1;
let colorDisplayTimeout;

// Calculate position on circle
function getPosition(nodeIndex) {
    const angle = - ((nodeIndex / n_nodes) * 2 * Math.PI); 
    const x = (circleRadius + arrowSize * 0.0) + (circleRadius + (arrowSize * 0.6)) * Math.cos(angle);
    const y = (circleRadius + arrowSize * 0.8) + (circleRadius + (arrowSize * 0.8)) * Math.sin(angle);

    return { x, y, angle };
}

// Create arrow element
function createArrow(fromNode, toNode, isNext = false) {

    console.log(`Creating arrow from ${fromNode} to ${toNode}, isNext: ${isNext}`);

    const toPos = getPosition(toNode);
    const angle = toPos.angle * (180 / Math.PI);
    
    const arrow = document.createElement('div');
    arrow.className = 'arrow' + (isNext ? ' next' : '');
    
    // // If next arrow, shrink by 50%
    // asize = isNext ? arrowSize * 0.5 : arrowSize;
    asize = arrowSize;

    // Arrow styling
    const color = toNode % 2 === 0 ? evenArrowColor : oddArrowColor;
    arrow.style.borderLeft = `${asize}px solid transparent`;
    arrow.style.borderRight = `${asize}px solid transparent`;
    arrow.style.borderBottom = `${asize * 1.5}px solid ${color}`;
    
    // Position arrow at the edge of the circle pointing inward
    arrow.style.left = `${toPos.x - asize}px`;
    arrow.style.top = `${toPos.y - asize * 1.5}px`;
    arrow.style.transform = `rotate(${angle - 90}deg)`;
    
    return arrow;
}

// Update counters
function updateCounters(index) {
    if (index < 0 || index >= orderedLines.length) return;
    
    const colorIdx = colorIndices[index];
    const sliceIdx = sliceIndices[index];
    
    // Count lines of this color we've seen so far
    let colorLinesSoFar = 0;
    for (let i = 0; i <= index; i++) {
        if (colorIndices[i] === colorIdx) colorLinesSoFar++;
    }
    
    // Count lines in this slice we've seen so far
    let sliceLinesSoFar = 0;
    for (let i = 0; i <= index; i++) {
        if (sliceIndices[i] === sliceIdx) sliceLinesSoFar++;
    }
    
    document.getElementById('group-counter').textContent = `Group ${sliceLinesSoFar}/${sliceLineCounts[sliceIdx]}`;
    document.getElementById('color-counter').textContent = `Color ${colorLinesSoFar}/${colorLineCounts[colorIdx]}`;
    document.getElementById('total-counter').textContent = `Total ${index + 1}/${totalLines}`;
}

// Show color name
function showColorName(index) {
    if (index < 0 || index >= orderedLines.length) return;
    
    const colorDisplay = document.getElementById('color-display');
    
    // Check if color changed
    if (index === 0 || colorIndices[index] !== colorIndices[index - 1]) {
        const colorName = colorStrings[colorIndices[index]];
        colorDisplay.textContent = colorName;
        colorDisplay.style.opacity = 1;
        
        // Clear any existing timeout
        if (colorDisplayTimeout) clearTimeout(colorDisplayTimeout);
        
        if (!keepColorName) {
            // Set timeout to hide after 10 lines
            let linesWithSameColor = 0;
            for (let i = index; i < orderedLines.length; i++) {
                if (colorIndices[i] === colorIndices[index]) linesWithSameColor++;
                else break;
            }
            
            const linesToShow = Math.min(10, linesWithSameColor);
            colorDisplayTimeout = setTimeout(() => {
                colorDisplay.style.opacity = 0;
            }, linesToShow * 1000); // Assuming ~1 second per line
        }
    }
}

// Advance visualization
function advance() {
    const container = document.getElementById('visualization-container');
    
    // Remove current arrow
    const currentArrow = document.querySelector('.arrow:not(.next)');
    if (currentArrow) container.removeChild(currentArrow);
    
    // Current next arrow becomes current
    const nextArrow = document.querySelector('.arrow.next');
    if (nextArrow) {
        nextArrow.classList.remove('next');
    }
    
    // Increment index
    currentLineIndex++;
    if (currentLineIndex >= orderedLines.length) {
        currentLineIndex = 0; // Loop back to beginning
    }
    
    // Update display
    updateCounters(currentLineIndex);
    showColorName(currentLineIndex);
    
    // Add next arrow
    if (currentLineIndex + 1 < orderedLines.length) {
        const nextLine = orderedLines[currentLineIndex + 1];
        container.appendChild(createArrow(nextLine[0], nextLine[1], true));
    }
}

// Retreat visualization
function retreat() {
    const container = document.getElementById('visualization-container');
    
    // Remove current arrow and next arrow
    const currentArrow = document.querySelector('.arrow:not(.next)');
    if (currentArrow) container.removeChild(currentArrow);
    
    const nextArrow = document.querySelector('.arrow.next');
    if (nextArrow) container.removeChild(nextArrow);
    
    // Decrement index
    currentLineIndex--;
    if (currentLineIndex < 0) {
        currentLineIndex = orderedLines.length - 1; // Loop back to end
    }
    
    // Update display
    updateCounters(currentLineIndex);
    showColorName(currentLineIndex);
    
    // Add current arrow
    const currentLine = orderedLines[currentLineIndex];
    container.appendChild(createArrow(currentLine[0], currentLine[1], false));
    
    // Add next arrow
    const nextIndex = (currentLineIndex + 1) % orderedLines.length;
    const nextLine = orderedLines[nextIndex];
    container.appendChild(createArrow(nextLine[0], nextLine[1], true));
}

// Initialize
function initialize() {
    const container = document.getElementById('visualization-container');
    const circle = document.getElementById('circle');
    
    // Set circle size
    circle.style.width = `${circleRadius * 2}px`;
    circle.style.height = `${circleRadius * 2}px`;
    
    // Center container
    container.style.width = `${circleRadius * 2}px`;
    container.style.height = `${circleRadius * 2}px`;
    
    // Add first two arrows
    if (orderedLines.length > 0) {
        const firstLine = orderedLines[0];
        container.appendChild(createArrow(firstLine[0], firstLine[1], true));
        advance(); // Show first arrow and set up the next one
    }
    
    // Add a small dash at the right-most point of the circle
    const dash = document.createElement('div');
    dash.style.position = 'absolute';
    dash.style.width = '10px';
    dash.style.height = '4px';
    dash.style.backgroundColor = evenArrowColor;
    dash.style.left = `${circleRadius * 2 - 3}px`; // Right-most point
    dash.style.top = `${circleRadius - 5}px`; // Centered vertically
    container.appendChild(dash);
    
    // Event listeners
    document.addEventListener('click', advance);
    document.addEventListener('wheel', (e) => {
        if (e.deltaY < 0) {
            // Scroll up - move backwards
            retreat();
        } else {
            // Scroll down - move forwards
            advance();
        }
    });
}

// Start when page loads
window.addEventListener('load', initialize);