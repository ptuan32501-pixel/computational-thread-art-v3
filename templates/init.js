const xPerm = randPerm * fullWidth;
const yPerm = randPerm * fullHeight;
const xPermSmall = randPerm * smallWidth;
const yPermSmall = randPerm * smallHeight;        

const margin = 0;
const frameDuration = 25;

// Setup full-color SVG and scales
const fullSvg = d3.select("#full-canvas");
fullSvg
    .attr("width", fullWidth + "px")
    .attr("height", fullHeight + "px")
    .style("background-color", bgColor);

const fullScaleX = d3.scaleLinear()
    .domain([0, d3.max(Object.values(data.d_coords), d => d[1])])
    .range([margin, fullWidth - margin]);
    
const fullScaleY = d3.scaleLinear()
    .domain([0, d3.max(Object.values(data.d_coords), d => d[0])])
    .range([margin, fullHeight - margin]);

// Create containers for individual color SVGs
const colorsContainer = d3.select(".colors-container");

// Generate individual color containers, and set up scales for them
data.palette.forEach((colorRGB, i) => {
    const rgbValues = colorRGB.match(/\d+/g).map(Number);
    const isVeryLightColor = rgbValues[0] + rgbValues[1] + rgbValues[2] >= 255 * 1.9;
    const backgroundColor = isVeryLightColor ? "black" : "white";

    // Create a container for each color
    const colorContainer = colorsContainer.append("div")
        .attr("class", "plot-container");

    // Add title
    colorContainer.append("div")
        .append("span")
        .attr("class", "plot-title")
        .text(colorNames && colorNames.length > 0 ? colorNames[i] : colorRGB);

    // Add SVG for this color
    colorContainer.append("svg")
        .attr("id", "canvas-" + i)
        .attr("class", "canvas")
        .attr("width", smallWidth + "px")
        .attr("height", smallHeight + "px")
        .style("background-color", backgroundColor);
});

// TODO - combine this code with some of the code above?
// Setup individual color SVGs (different for each color) and scales
const svgs = {};
data.palette.forEach((color, i) => {
    svgs[color] = d3.select("#canvas-" + i);
});

scaleX = d3.scaleLinear()
    .domain([0, d3.max(Object.values(data.d_coords), d => d[1])])
    .range([margin/2, smallWidth - margin/2]);
    
scaleY = d3.scaleLinear()
    .domain([0, d3.max(Object.values(data.d_coords), d => d[0])])
    .range([margin/2, smallHeight - margin/2]);
    

// Function for slicing list
function getSlice(arr, x, y, color) {
    let start = Math.floor((x / y) * arr.length);
    let end = Math.floor(((x + 1) / y) * arr.length);
    return arr.slice(start, end).map(coords => ({ coords, color }));
}

// Create separate line lists for each color
let colorLines = {};
let allLines = [];

data.palette.forEach(color => {
    colorLines[color] = [];
});

// Process lines according to group orders and separate by color
data.group_orders_list.forEach((groupIdx, step) => {
    let color = data.palette[groupIdx];
    let lines = getSlice(data.line_dict[color], data.group_orders_count[groupIdx], data.group_orders_total[groupIdx], color);
    data.group_orders_count[groupIdx]++;
    colorLines[color].push(...lines);
    allLines.push(...lines);
    console.log(`Group ${groupIdx} (${color}) - Step ${step}: ${lines.length} lines`);
});

document.getElementById("slider").max = 1;

// Generate line data with unique IDs for tracking
const allLinesWithIds = allLines.map((line, index) => ({
    ...line,
    id: `line-${index}`
}));

const colorLinesWithIds = {};
data.palette.forEach(color => {
    colorLinesWithIds[color] = colorLines[color].map((line, index) => ({
        ...line,
        id: `${color}-line-${index}`
    }));
});

// Track currently displayed lines
let currentFullLines = [];
let currentColorLines = {};
data.palette.forEach(color => {
    currentColorLines[color] = [];
});

// Create line elements with their attributes but keep them hidden initially
function createAllLineElements() {
    // Create all lines for the full canvas
    fullSvg.selectAll("line")
        .data(allLinesWithIds, d => d.id)
        .enter()
        .append("line")
        .attr("x1", d => fullScaleX(data.d_coords[d.coords[0]][1]) + (Math.random() * 2 * xPerm - xPerm))
        .attr("y1", d => fullScaleY(data.d_coords[d.coords[0]][0]) + (Math.random() * 2 * yPerm - yPerm))
        .attr("x2", d => fullScaleX(data.d_coords[d.coords[1]][1]) + (Math.random() * 2 * xPerm - xPerm))
        .attr("y2", d => fullScaleY(data.d_coords[d.coords[1]][0]) + (Math.random() * 2 * yPerm - yPerm))
        .attr("stroke", d => d.color)
        .attr("stroke-width", lineWidth)
        .attr("visibility", "hidden") // Start with all lines hidden
        .attr("data-index", (d, i) => i); // Store index for easy access
    
    // Create all lines for each color canvas
    data.palette.forEach(color => {
        svgs[color].selectAll("line")
            .data(colorLinesWithIds[color], d => d.id)
            .enter()
            .append("line")
            .attr("x1", d => scaleX(data.d_coords[d.coords[0]][1]) + (Math.random() * 2 * xPermSmall - xPermSmall))
            .attr("y1", d => scaleY(data.d_coords[d.coords[0]][0]) + (Math.random() * 2 * yPermSmall - yPermSmall))
            .attr("x2", d => scaleX(data.d_coords[d.coords[1]][1]) + (Math.random() * 2 * xPermSmall - xPermSmall))
            .attr("y2", d => scaleY(data.d_coords[d.coords[1]][0]) + (Math.random() * 2 * yPermSmall - yPermSmall))
            .attr("stroke", d => d.color)
            .attr("stroke-width", lineWidth/2)
            .attr("visibility", "hidden") // Start with all lines hidden
            .attr("data-index", (d, i) => i); // Store index for easy access
    });
}

// Call once to create all line elements
createAllLineElements();

function updateVisibility(progress) {
    // Calculate how many lines should be visible
    const totalLinesToShow = Math.ceil(progress * allLinesWithIds.length);
    
    // For the full canvas
    fullSvg.selectAll("line")
        .attr("visibility", (d, i) => i < totalLinesToShow ? "visible" : "hidden");
    
    // For each color canvas
    data.palette.forEach(color => {
        const totalLinesForColor = colorLinesWithIds[color].length;
        const linesToShowForColor = Math.ceil(progress * totalLinesForColor);
        
        svgs[color].selectAll("line")
            .attr("visibility", (d, i) => i < linesToShowForColor ? "visible" : "hidden");
    });
}

// Initialize animation control variables
let animationInterval = null;
let isPlaying = false;
const totalSteps = nSteps;
let currentStep = 0;

// Play/pause functionality
const playPauseBtn = document.getElementById('play-pause-btn');
const playIcon = document.getElementById('play-icon');
const pauseIcon = document.getElementById('pause-icon');

playPauseBtn.addEventListener('click', function() {
    if (isPlaying) {
        pauseAnimation();
    } else {
        playAnimation();
    }
});

// Reset button functionality
const resetBtn = document.getElementById('reset-btn');
resetBtn.addEventListener('click', function() {
    resetAnimation();
});

function playAnimation() {
    if (isPlaying) return;
    
    isPlaying = true;
    playIcon.style.display = 'none';
    pauseIcon.style.display = 'block';
    
    // If we reached the end, start from beginning
    if (parseFloat(slider.value) >= 1) {
        slider.value = 0;
        currentStep = 0;
    }
    
    animationInterval = setInterval(() => {
        if (currentStep > totalSteps) {
            pauseAnimation();
            return;
        }
        
        const progress = currentStep / totalSteps;
        slider.value = progress;
        updateVisibility(progress);
        currentStep++;
    }, frameDuration);
}

function pauseAnimation() {
    if (!isPlaying) return;
    
    isPlaying = false;
    clearInterval(animationInterval);
    playIcon.style.display = 'block';
    pauseIcon.style.display = 'none';
}

function resetAnimation() {
    pauseAnimation();
    currentStep = 0;
    slider.value = 0;
    updateVisibility(0);
}

// Update the existing animate function to work with our new controls
function animate() {
    // Start the animation automatically
    playAnimation();
}

// Modify the slider event listener to update the step counter
slider.addEventListener("input", function() {
    const currentValue = parseFloat(this.value);
    updateVisibility(currentValue);
    // Update the step counter based on slider position
    currentStep = Math.round(currentValue * totalSteps);
    previousValue = currentValue;
    
    // If manually moved to the end, pause the animation
    if (currentValue >= 1) {
        pauseAnimation();
    }
});

function animate() {
    let step = 0;
    const totalSteps = nSteps;
    let interval = setInterval(() => {
        if (step > totalSteps) {
            clearInterval(interval);
            return;
        }
        const progress = step / totalSteps;
        slider.value = progress;
        updateVisibility(progress);
        step++;
    }, frameDuration);
}

// Start the animation
animate();