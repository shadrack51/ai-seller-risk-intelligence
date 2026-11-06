"use strict";

// ============================================================
// APPLICATION CONFIGURATION
// ============================================================

const API_BASE_URL = "http://127.0.0.1:8000";
const DEBUG_MODE = false;

let currentSellerIntelligence = null;


// ============================================================
// DEBUG LOGGER
// ============================================================

function debugLog(...messages) {
    if (DEBUG_MODE) {
        console.log(...messages);
    }
}


// ============================================================
// DOM ELEMENTS
// ============================================================

const sellerForm = document.getElementById("seller-form");
const sellerInput = document.getElementById("seller-id");
const assessButton = document.getElementById("assess-button");
const apiStatus = document.getElementById("api-status");
const formError = document.getElementById("form-error");
const resultContainer = document.getElementById("result-container");


// ============================================================
// DOM VALIDATION
// ============================================================

function validateDomElements() {
    const requiredElements = {
        "seller-form": sellerForm,
        "seller-id": sellerInput,
        "assess-button": assessButton,
        "api-status": apiStatus,
        "form-error": formError,
        "result-container": resultContainer
    };

    const missingElements = Object.entries(requiredElements)
        .filter(([, element]) => !element)
        .map(([name]) => name);

    if (missingElements.length > 0) {
        console.error("Frontend DOM validation failed.", missingElements);
        return false;
    }

    debugLog("Frontend DOM validation successful.");
    return true;
}

const domIsValid = validateDomElements();


// ============================================================
// API STATUS
// ============================================================

async function checkApiStatus() {
    if (!apiStatus) return;

    apiStatus.textContent = "Checking API...";

    try {
        const response = await fetch(`${API_BASE_URL}/health`);

        if (!response.ok) {
            throw new Error(`Health endpoint returned HTTP ${response.status}.`);
        }

        const data = await response.json();
        debugLog("FastAPI health response:", data);

        apiStatus.textContent = "API Online";
        apiStatus.classList.add("online");
        apiStatus.classList.remove("offline");
    } catch (error) {
        console.error("FastAPI health check failed:", error);

        apiStatus.textContent = "API Offline";
        apiStatus.classList.remove("online");
        apiStatus.classList.add("offline");
    }
}


// ============================================================
// FORM ERROR HANDLING
// ============================================================

function showFormError(message) {
    if (!formError) return;

    formError.textContent = message;
    formError.classList.remove("hidden");
}

function clearFormError() {
    if (!formError) return;

    formError.textContent = "";
    formError.classList.add("hidden");
}


// ============================================================
// BUTTON / LOADING STATE
// ============================================================

function setLoadingState(isLoading) {
    if (!assessButton) return;

    assessButton.disabled = isLoading;
    assessButton.textContent = isLoading
        ? "Assessing Seller..."
        : "Assess Seller";
}


// ============================================================
// FORMATTERS
// ============================================================

function numericValue(value) {
    if (value === null || value === undefined || value === "") {
        return null;
    }

    const number = Number(value);

    return Number.isFinite(number) ? number : null;
}

function formatNumber(value) {
    const number = numericValue(value);

    if (number === null) return "—";

    return number.toLocaleString(undefined, {
        maximumFractionDigits: 2
    });
}

function formatCurrency(value) {
    const number = numericValue(value);

    if (number === null) return "—";

    return number.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
}

function formatPercentage(value) {
    const number = numericValue(value);

    if (number === null) return "—";

    return `${(number * 100).toFixed(1)}%`;
}

function formatContribution(value) {
    const number = numericValue(value);

    if (number === null) return "—";

    return `${(Math.abs(number) * 100).toFixed(1)}%`;
}

function formatDate(value) {
    if (!value) return "—";

    const date = new Date(value);

    if (Number.isNaN(date.getTime())) {
        return String(value);
    }

    return date.toLocaleDateString(undefined, {
        year: "numeric",
        month: "short",
        day: "numeric"
    });
}


// ============================================================
// HTML SAFETY
// ============================================================

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


// ============================================================
// THRESHOLD RESOLUTION
// ============================================================

function getThreshold(assessment, preferredField) {
    if (!assessment) return NaN;

    const preferredValue = numericValue(assessment[preferredField]);

    if (preferredValue !== null) {
        return preferredValue;
    }

    const genericValue = numericValue(assessment.threshold);

    if (genericValue !== null) {
        return genericValue;
    }

    console.warn(
        `No valid threshold found. Expected "${preferredField}" or "threshold".`,
        assessment
    );

    return NaN;
}


// ============================================================
// RISK STATUS
// ============================================================

function getRiskStatus(riskScore, threshold, explicitStatus) {
    if (typeof explicitStatus === "boolean") {
        return explicitStatus;
    }

    const score = numericValue(riskScore);
    const operatingThreshold = numericValue(threshold);

    if (score === null || operatingThreshold === null) {
        return null;
    }

    return score >= operatingThreshold;
}

function getStatusLabel(elevated) {
    if (elevated === null || elevated === undefined) {
        return "STATUS UNAVAILABLE";
    }

    return elevated ? "ABOVE THRESHOLD" : "WITHIN THRESHOLD";
}

function getRiskBadgeClass(elevated) {
    if (elevated === null || elevated === undefined) {
        return "risk-unavailable";
    }

    return elevated ? "risk-elevated" : "risk-within";
}

function getMeterWidth(riskScore) {
    const score = numericValue(riskScore);

    if (score === null) return 0;

    return Math.min(Math.max((score * 100 / 50) * 100, 0), 100);
}


// ============================================================
// MODEL DRIVER CARDS
// ============================================================

function createDriverCard(driver, index) {
    const direction = driver.direction || "neutral";

    let directionSymbol = "→";
    let directionClass = "driver-neutral";
    let directionText = "Neutral";

    if (direction === "increases_risk") {
        directionSymbol = "↑";
        directionClass = "driver-risk";
        directionText = "Increases risk";
    } else if (direction === "reduces_risk") {
        directionSymbol = "↓";
        directionClass = "driver-reduces";
        directionText = "Reduces risk";
    }

    const featureLabel =
        driver.feature_label || driver.feature || "Model feature";

    const value = driver.display_value ?? driver.value ?? null;
    const contribution = driver.contribution ?? driver.impact ?? null;

    return `
        <article class="driver-card">
            <div class="driver-rank">
                ${String(index + 1).padStart(2, "0")}
            </div>

            <div class="driver-main">
                <div class="driver-title">
                    <span class="${directionClass}">
                        ${directionSymbol}
                    </span>
                    ${escapeHtml(featureLabel)}
                </div>

                <div class="driver-details">
                    ${
                        value !== null
                            ? `<span>Value: <strong>${escapeHtml(formatNumber(value))}</strong></span>`
                            : ""
                    }

                    <span class="${directionClass}">
                        ${directionText}
                    </span>
                </div>
            </div>

            <div class="driver-contribution">
                ${contribution !== null ? formatContribution(contribution) : "—"}
            </div>
        </article>
    `;
}


function createDriverSection(title, assessment, modelType) {
    if (!assessment) return "";

    const drivers = Array.isArray(assessment.drivers)
        ? assessment.drivers
        : [];

    const riskScore = numericValue(assessment.risk_score);

    const recommendedAction =
        assessment.recommended_action ||
        "Continue monitoring this seller.";

    const driverCards = drivers.length > 0
        ? drivers.map(createDriverCard).join("")
        : `
            <div class="no-drivers">
                <strong>Driver explanations unavailable</strong>
                <p>
                    The prediction was returned successfully,
                    but no local explanation data was supplied by the API.
                </p>
            </div>
        `;

    return `
        <section class="explanation-section">
            <div class="explanation-section-header">
                <div>
                    <div class="explanation-kicker">
                        ${escapeHtml(modelType)}
                    </div>

                    <h3>${escapeHtml(title)}</h3>
                </div>

                <div class="explanation-score">
                    ${formatPercentage(riskScore)}
                </div>
            </div>

            <div class="drivers-heading">Top model drivers</div>

            <div class="drivers-list">
                ${driverCards}
            </div>

            <div class="action-box">
                <div class="action-label">Recommended Action</div>
                <p>${escapeHtml(recommendedAction)}</p>
            </div>
        </section>
    `;
}


// ============================================================
// DISPLAY RISK RESULT
// ============================================================

function displayRiskResult(data) {
    if (!resultContainer) {
        console.error("Result container not found.");
        return false;
    }

    const delivery = data.delivery;
    const review = data.review;

    if (!delivery || !review) {
        showFormError("The API returned an incomplete seller assessment.");
        console.error("Incomplete seller assessment:", data);
        return false;
    }

    const sellerId = data.seller_id || "Unknown seller";
    const snapshotDate = data.snapshot_date || "Unknown date";

    const deliveryRisk = numericValue(delivery.risk_score);
    const deliveryThreshold = getThreshold(delivery, "monitoring_threshold");

    const deliveryElevated = getRiskStatus(
        deliveryRisk,
        deliveryThreshold,
        delivery.monitor_delivery
    );

    const reviewRisk = numericValue(review.risk_score);
    const reviewThreshold = getThreshold(review, "screening_threshold");

    const reviewElevated = getRiskStatus(
        reviewRisk,
        reviewThreshold,
        review.screen_for_review
    );

    resultContainer.innerHTML = `
        <div class="seller-result-header">
            <div>
                <div class="result-label">RISK RESULT</div>
                <h2>Seller Assessment</h2>
            </div>

            <div class="seller-result-meta">
                <div class="result-label">SELLER</div>
                <strong>${escapeHtml(sellerId)}</strong>
                <div class="snapshot">
                    Snapshot ${escapeHtml(snapshotDate)}
                </div>
            </div>
        </div>

        <div class="risk-results">
            <section class="risk-item">
                <div class="risk-item-header">
                    <div class="result-label">DELIVERY RISK</div>

                    <div class="risk-badge ${getRiskBadgeClass(deliveryElevated)}">
                        ${getStatusLabel(deliveryElevated)}
                    </div>
                </div>

                <div class="risk-score">
                    ${formatPercentage(deliveryRisk)}
                </div>

                <div class="risk-meta">
                    Threshold:
                    <strong>${formatPercentage(deliveryThreshold)}</strong>
                </div>

                <div class="risk-meter">
                    <div
                        class="risk-meter-fill"
                        style="width: ${getMeterWidth(deliveryRisk)}%;"
                    ></div>
                </div>

                <div class="risk-scale">
                    <span>0%</span>
                    <span>50%+</span>
                </div>

                <div class="risk-model">
                    <span>Model</span>
                    <strong>
                        ${escapeHtml(delivery.model || "RandomForestClassifier")}
                    </strong>
                </div>
            </section>

            <section class="risk-item">
                <div class="risk-item-header">
                    <div class="result-label">REVIEW RISK</div>

                    <div class="risk-badge ${getRiskBadgeClass(reviewElevated)}">
                        ${getStatusLabel(reviewElevated)}
                    </div>
                </div>

                <div class="risk-score">
                    ${formatPercentage(reviewRisk)}
                </div>

                <div class="risk-meta">
                    Threshold:
                    <strong>${formatPercentage(reviewThreshold)}</strong>
                </div>

                <div class="risk-meter">
                    <div
                        class="risk-meter-fill"
                        style="width: ${getMeterWidth(reviewRisk)}%;"
                    ></div>
                </div>

                <div class="risk-scale">
                    <span>0%</span>
                    <span>50%+</span>
                </div>

                <div class="risk-model">
                    <span>Model</span>
                    <strong>
                        ${escapeHtml(review.model || "LogisticRegression")}
                    </strong>
                </div>
            </section>
        </div>

        <section class="model-explainability">
            <div class="result-label">MODEL EXPLAINABILITY</div>

            <h2>Why did the models reach this assessment?</h2>

            <p class="explanation-intro">
                The following features contributed most strongly
                to each model's prediction for this seller.
            </p>

            <div class="explanation-grid">
                ${createDriverSection(
                    "Why is delivery risk elevated?",
                    delivery,
                    "DELIVERY MODEL"
                )}

                ${createDriverSection(
                    "Why is review risk elevated?",
                    review,
                    "REVIEW MODEL"
                )}
            </div>
        </section>
    `;

    return true;
}


// ============================================================
// SELLER INTELLIGENCE HELPERS
// ============================================================

function getLatestSnapshot(intelligence) {
    if (
        !intelligence ||
        !Array.isArray(intelligence.snapshots) ||
        intelligence.snapshots.length === 0
    ) {
        return null;
    }

    // The API's snapshots are ordered oldest to newest.
    return intelligence.snapshots[intelligence.snapshots.length - 1];
}


function getTrendClass(value) {
    const number = numericValue(value);

    if (number === null || number === 0) {
        return "intelligence-neutral";
    }

    return number > 0
        ? "intelligence-positive"
        : "intelligence-negative";
}


function getTrendSymbol(value) {
    const number = numericValue(value);

    if (number === null) return "—";
    if (number > 0) return "↑";
    if (number < 0) return "↓";

    return "→";
}


function createIntelligenceKpi(label, value, change, formatter) {
    return `
        <article class="intelligence-kpi">
            <div class="intelligence-kpi-label">
                ${escapeHtml(label)}
            </div>

            <div class="intelligence-kpi-value">
                ${escapeHtml(formatter(value))}
            </div>

            <div class="intelligence-kpi-change ${getTrendClass(change)}">
                <span>${getTrendSymbol(change)}</span>
                <span>${escapeHtml(formatter(change))}</span>
            </div>
        </article>
    `;
}


// ============================================================
// HISTORICAL TABLE ROW
// ============================================================

function createHistoricalSnapshotRow(snapshot) {
    const deliveryDays = numericValue(
        snapshot.recent_average_delivery_days
    );

    const lateRate = numericValue(
        snapshot.recent_late_delivery_rate
    );

    const reviewScore = numericValue(
        snapshot.recent_average_review_score
    );

    return `
        <tr>
            <td>${escapeHtml(formatDate(snapshot.snapshot_date))}</td>
            <td>${escapeHtml(formatNumber(snapshot.recent_orders))}</td>
            <td>${escapeHtml(formatNumber(snapshot.recent_units_sold))}</td>
            <td>${escapeHtml(formatCurrency(snapshot.recent_revenue))}</td>
            <td>
                ${deliveryDays === null
                    ? "—"
                    : `${escapeHtml(formatNumber(deliveryDays))} days`}
            </td>
            <td>
                ${lateRate === null
                    ? "—"
                    : escapeHtml(formatPercentage(lateRate))}
            </td>
            <td>
                ${reviewScore === null
                    ? "—"
                    : escapeHtml(formatNumber(reviewScore))}
            </td>
        </tr>
    `;
}


// ============================================================
// HISTORICAL TREND CHARTS
// Native SVG — no external chart library required
// ============================================================

function createTrendChart(snapshots, configuration) {
    const {
        key,
        title,
        description,
        formatter
    } = configuration;

    const chartWidth = 600;
    const chartHeight = 250;

    const paddingLeft = 56;
    const paddingRight = 22;
    const paddingTop = 22;
    const paddingBottom = 42;

    const plotWidth = chartWidth - paddingLeft - paddingRight;
    const plotHeight = chartHeight - paddingTop - paddingBottom;

    const points = snapshots.map((snapshot, index) => ({
        index,
        date: snapshot.snapshot_date,
        value: numericValue(snapshot[key])
    }));

    const validPoints = points.filter(point => point.value !== null);

    if (validPoints.length === 0) {
        return `
            <article class="trend-chart-card">
                <h3>${escapeHtml(title)}</h3>
                <p>${escapeHtml(description)}</p>
                <div class="trend-chart-empty">
                    No historical values are available for this metric.
                </div>
            </article>
        `;
    }

    let minimum = Math.min(...validPoints.map(point => point.value));
    let maximum = Math.max(...validPoints.map(point => point.value));

    // Give constant-value series some vertical breathing room.
    if (minimum === maximum) {
        const padding = Math.abs(minimum) * 0.1 || 1;
        minimum -= padding;
        maximum += padding;
    } else {
        const padding = (maximum - minimum) * 0.12;
        minimum -= padding;
        maximum += padding;
    }

    const xPosition = index => {
        if (points.length <= 1) {
            return paddingLeft + plotWidth / 2;
        }

        return paddingLeft + (index / (points.length - 1)) * plotWidth;
    };

    const yPosition = value =>
        paddingTop +
        ((maximum - value) / (maximum - minimum)) * plotHeight;

    const gridLines = Array.from({ length: 5 }, (_, index) => {
        const fraction = index / 4;
        const value = maximum - fraction * (maximum - minimum);
        const y = paddingTop + fraction * plotHeight;

        return `
            <line
                x1="${paddingLeft}"
                y1="${y}"
                x2="${chartWidth - paddingRight}"
                y2="${y}"
                stroke="#24364d"
                stroke-width="1"
            />

            <text
                x="${paddingLeft - 9}"
                y="${y + 4}"
                fill="#71869f"
                font-size="10"
                text-anchor="end"
            >${escapeHtml(formatter(value))}</text>
        `;
    }).join("");

    const linePath = validPoints
        .map((point, index) => {
            const command = index === 0 ? "M" : "L";

            return `${command} ${xPosition(point.index)} ${yPosition(point.value)}`;
        })
        .join(" ");

    const areaPath = validPoints
        .map((point, index) => {
            const command = index === 0 ? "M" : "L";

            return `${command} ${xPosition(point.index)} ${yPosition(point.value)}`;
        })
        .join(" ");

    const firstValid = validPoints[0];
    const lastValid = validPoints[validPoints.length - 1];

    const filledArea = validPoints.length > 1
        ? `
            <path
                d="${areaPath}
                    L ${xPosition(lastValid.index)} ${paddingTop + plotHeight}
                    L ${xPosition(firstValid.index)} ${paddingTop + plotHeight}
                    Z"
                fill="#60a5fa"
                fill-opacity="0.08"
            />
        `
        : "";

    const dots = validPoints.map(point => `
        <circle
            cx="${xPosition(point.index)}"
            cy="${yPosition(point.value)}"
            r="3.5"
            fill="#60a5fa"
            stroke="#0b1522"
            stroke-width="2"
        >
            <title>
                ${escapeHtml(formatDate(point.date))}:
                ${escapeHtml(formatter(point.value))}
            </title>
        </circle>
    `).join("");

    // Show at most five date labels to avoid overcrowding.
    const labelIndices = new Set();

    if (points.length > 0) {
        labelIndices.add(0);
        labelIndices.add(points.length - 1);

        if (points.length > 2) {
            labelIndices.add(Math.floor((points.length - 1) / 2));
        }

        if (points.length > 6) {
            labelIndices.add(Math.floor((points.length - 1) / 4));
            labelIndices.add(Math.floor(((points.length - 1) * 3) / 4));
        }
    }

    const dateLabels = points
        .filter(point => labelIndices.has(point.index))
        .map(point => {
            const label = point.date
                ? point.date.slice(0, 7)
                : "Unknown";

            return `
                <text
                    x="${xPosition(point.index)}"
                    y="${chartHeight - 13}"
                    fill="#71869f"
                    font-size="10"
                    text-anchor="middle"
                >${escapeHtml(label)}</text>
            `;
        })
        .join("");

    const latestValue = lastValid.value;
    const earliestValue = firstValid.value;
    const netChange = latestValue - earliestValue;

    return `
        <article class="trend-chart-card">
            <div class="trend-chart-heading">
                <div>
                    <h3>${escapeHtml(title)}</h3>
                    <p>${escapeHtml(description)}</p>
                </div>

                <div class="trend-chart-latest">
                    <span>Latest</span>
                    <strong>${escapeHtml(formatter(latestValue))}</strong>
                </div>
            </div>

            <div class="trend-chart-visual">
                <svg
                    viewBox="0 0 ${chartWidth} ${chartHeight}"
                    role="img"
                    aria-label="${escapeHtml(title)} historical trend chart"
                    preserveAspectRatio="xMidYMid meet"
                >
                    ${gridLines}
                    ${filledArea}

                    <path
                        d="${linePath}"
                        fill="none"
                        stroke="#60a5fa"
                        stroke-width="2.5"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    />

                    ${dots}
                    ${dateLabels}
                </svg>
            </div>

            <div class="trend-chart-summary">
                <span>
                    First available: ${escapeHtml(formatter(earliestValue))}
                </span>

                <span class="${getTrendClass(netChange)}">
                    Net change: ${getTrendSymbol(netChange)}
                    ${escapeHtml(formatter(netChange))}
                </span>
            </div>
        </article>
    `;
}


// ============================================================
// RENDER HISTORICAL TREND VISUALIZATION
// ============================================================

function renderHistoricalCharts(snapshots) {
    const chartConfigurations = [
        {
            key: "recent_orders",
            title: "Orders",
            description: "Recent order activity at each snapshot.",
            formatter: formatNumber
        },
        {
            key: "recent_units_sold",
            title: "Units Sold",
            description: "Recent units sold at each snapshot.",
            formatter: formatNumber
        },
        {
            key: "recent_revenue",
            title: "Revenue",
            description: "Recent revenue at each snapshot.",
            formatter: formatCurrency
        },
        {
            key: "recent_average_delivery_days",
            title: "Delivery Time",
            description: "Average delivery duration, measured in days.",
            formatter: value => {
                const number = numericValue(value);

                return number === null
                    ? "—"
                    : `${formatNumber(number)} days`;
            }
        },
        {
            key: "recent_average_review_score",
            title: "Review Score",
            description: "Average customer review score over time.",
            formatter: formatNumber
        }
    ];

    const charts = chartConfigurations
        .map(configuration => createTrendChart(snapshots, configuration))
        .join("");

    return `
        <section class="historical-trends">
            <div class="historical-trends-header">
                <div class="result-label">HISTORICAL ANALYTICS</div>

                <h3>How has this seller performed over time?</h3>

                <p>
                    Each point represents a historical snapshot.
                    These charts describe observed business behaviour;
                    they do not represent future predictions.
                </p>
            </div>

            <div class="historical-trends-grid">
                ${charts}
            </div>
        </section>
    `;
}


// ============================================================
// RENDER SELLER INTELLIGENCE
// ============================================================

function renderSellerIntelligence(intelligence) {
    if (!resultContainer) return;

    if (
        !intelligence ||
        !Array.isArray(intelligence.snapshots) ||
        intelligence.snapshots.length === 0
    ) {
        return;
    }

    const snapshots = intelligence.snapshots;
    const latestSnapshot = getLatestSnapshot(intelligence);

    if (!latestSnapshot) return;

    const snapshotCount = snapshots.length;

    const historicalRows = snapshots
        .slice()
        .reverse()
        .map(createHistoricalSnapshotRow)
        .join("");

    const intelligenceSection = document.createElement("section");
    intelligenceSection.className = "seller-intelligence";

    intelligenceSection.innerHTML = `
        <div class="seller-intelligence-header">
            <div>
                <div class="result-label">SELLER INTELLIGENCE</div>

                <h2>Historical Business Behaviour</h2>

                <p class="intelligence-intro">
                    A historical view of this seller's activity,
                    commercial performance, delivery behaviour,
                    and customer feedback.
                </p>
            </div>

            <div class="intelligence-period">
                <span>HISTORICAL SNAPSHOTS</span>
                <strong>${escapeHtml(String(snapshotCount))}</strong>
            </div>
        </div>

        <div class="intelligence-latest">
            <div class="intelligence-latest-header">
                <div>
                    <div class="result-label">LATEST SNAPSHOT</div>

                    <h3>${escapeHtml(formatDate(latestSnapshot.snapshot_date))}</h3>
                </div>

                <p>
                    ${escapeHtml(String(snapshotCount))}
                    historical observations
                </p>
            </div>

            <div class="intelligence-kpi-grid">
                ${createIntelligenceKpi(
                    "Recent Orders",
                    latestSnapshot.recent_orders,
                    latestSnapshot.orders_change,
                    formatNumber
                )}

                ${createIntelligenceKpi(
                    "Units Sold",
                    latestSnapshot.recent_units_sold,
                    latestSnapshot.units_change,
                    formatNumber
                )}

                ${createIntelligenceKpi(
                    "Revenue",
                    latestSnapshot.recent_revenue,
                    latestSnapshot.revenue_change,
                    formatCurrency
                )}

                ${createIntelligenceKpi(
                    "Delivery Time",
                    latestSnapshot.recent_average_delivery_days,
                    latestSnapshot.delivery_days_change,
                    value => {
                        const number = numericValue(value);

                        return number === null
                            ? "—"
                            : `${formatNumber(number)} days`;
                    }
                )}

                ${createIntelligenceKpi(
                    "Late Delivery Rate",
                    latestSnapshot.recent_late_delivery_rate,
                    latestSnapshot.late_delivery_rate_change,
                    formatPercentage
                )}

                ${createIntelligenceKpi(
                    "Review Score",
                    latestSnapshot.recent_average_review_score,
                    latestSnapshot.review_score_change,
                    formatNumber
                )}

                ${createIntelligenceKpi(
                    "Low Review Rate",
                    latestSnapshot.recent_low_review_rate,
                    latestSnapshot.low_review_rate_change,
                    formatPercentage
                )}
            </div>
        </div>

        ${renderHistoricalCharts(snapshots)}

        <div class="intelligence-history">
            <div class="intelligence-history-section">
                <div class="intelligence-history-header">
                    <div>
                        <div class="result-label">HISTORICAL RECORDS</div>
                        <h3>Seller Performance by Snapshot</h3>
                    </div>

                    <div class="intelligence-history-note">
                        Recent-window values
                    </div>
                </div>

                <div class="intelligence-table-wrapper">
                    <table class="intelligence-table">
                        <thead>
                            <tr>
                                <th>Snapshot</th>
                                <th>Orders</th>
                                <th>Units</th>
                                <th>Revenue</th>
                                <th>Delivery</th>
                                <th>Late Rate</th>
                                <th>Review Score</th>
                            </tr>
                        </thead>

                        <tbody>${historicalRows}</tbody>
                    </table>
                </div>
            </div>
        </div>
    `;

    resultContainer.appendChild(intelligenceSection);
}


// ============================================================
// SELLER INTELLIGENCE REQUEST
// ============================================================

async function loadSellerIntelligence(sellerId) {
    const endpoint =
        `${API_BASE_URL}/intelligence/seller/` +
        encodeURIComponent(sellerId);

    debugLog("Loading Seller Intelligence:", endpoint);

    try {
        const response = await fetch(endpoint);

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error(
                "The Seller Intelligence API returned invalid JSON."
            );
        }

        if (!response.ok) {
            const detail = data.detail || data.message;

            throw new Error(
                typeof detail === "string"
                    ? detail
                    : "Unable to load Seller Intelligence."
            );
        }

        currentSellerIntelligence = data;

        debugLog("Seller Intelligence loaded successfully:", data);

        return data;
    } catch (error) {
        currentSellerIntelligence = null;

        console.error("Seller Intelligence failed:", error);

        return null;
    }
}


// ============================================================
// SELLER ASSESSMENT REQUEST
// ============================================================

async function assessSeller(sellerId) {
    debugLog("Starting seller assessment:", sellerId);

    clearFormError();
    setLoadingState(true);

    // Clear the previous seller intelligence before a new assessment.
    currentSellerIntelligence = null;

    try {
        const response = await fetch(`${API_BASE_URL}/assess/seller`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                seller_id: sellerId
            })
        });

        let data;

        try {
            data = await response.json();
        } catch {
            throw new Error("The API returned an invalid JSON response.");
        }

        if (!response.ok) {
            const detail = data.detail || data.message;

            throw new Error(
                typeof detail === "string"
                    ? detail
                    : "Seller assessment failed."
            );
        }

        // Step 1: Display the existing risk assessment.
        const assessmentDisplayed = displayRiskResult(data);

        if (!assessmentDisplayed) {
            return;
        }

        // Step 2: Retrieve historical seller intelligence.
        const intelligence = await loadSellerIntelligence(sellerId);

        // Step 3: Display KPIs, charts, and historical records.
        if (intelligence) {
            renderSellerIntelligence(intelligence);
        } else {
            showFormError(
                "Risk assessment succeeded, but historical Seller Intelligence could not be loaded."
            );
        }
    } catch (error) {
        console.error("Seller assessment failed:", error);

        showFormError(error.message || "Unable to assess seller.");
    } finally {
        setLoadingState(false);
    }
}


// ============================================================
// FORM SUBMISSION
// ============================================================

if (sellerForm) {
    sellerForm.addEventListener("submit", async event => {
        event.preventDefault();
        clearFormError();

        const sellerId = sellerInput.value.trim();

        if (!sellerId) {
            showFormError("Please enter a seller ID.");
            sellerInput.focus();
            return;
        }

        await assessSeller(sellerId);
    });
} else {
    console.error("Seller form was not found.");
}


// ============================================================
// APPLICATION INITIALIZATION
// ============================================================

if (domIsValid) {
    checkApiStatus();
} else {
    console.error("Frontend initialization stopped.");
}