// ─── Core Types ────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct Sensor {
    pub id: u8,
    pub value: f64,
    pub confidence: f64,
    pub weight: f64,
    pub bias: f64,
    pub timestamp: u64,
    pub active: bool,
}

#[derive(Clone, Debug)]
pub struct FusedSignal {
    pub value: f64,
    pub confidence: f64,
    pub variance: f64,
    pub source_count: u8,
    pub timestamp: u64,
}

pub struct Engine {
    sensors: Vec<Sensor>,
    history: Vec<FusedSignal>,
    threshold: f64,
}

impl Engine {
    pub fn new(threshold: f64) -> Self {
        Self {
            sensors: Vec::new(),
            history: Vec::new(),
            threshold,
        }
    }

    pub fn add_sensor(&mut self, id: u8, weight: f64, bias: f64) {
        self.sensors.push(Sensor {
            id,
            value: 0.0,
            confidence: 0.0,
            weight,
            bias,
            timestamp: 0,
            active: true,
        });
    }

    pub fn find_sensor(&self, id: u8) -> Option<&Sensor> {
        self.sensors.iter().find(|s| s.id == id)
    }

    pub fn update(&mut self, sensor_id: u8, value: f64, confidence: f64, now: u64) {
        if let Some(sensor) = self
            .sensors
            .iter_mut()
            .find(|s| s.id == sensor_id && s.active)
        {
            sensor.value = value;
            sensor.confidence = confidence;
            sensor.timestamp = now;
        }
    }

    pub fn read(&self) -> FusedSignal {
        let active: Vec<&Sensor> = self
            .sensors
            .iter()
            .filter(|s| s.active && s.confidence > self.threshold)
            .collect();

        if active.is_empty() {
            let ts = self.sensors.iter().map(|s| s.timestamp).max().unwrap_or(0);
            return FusedSignal {
                value: 0.0,
                confidence: 0.0,
                variance: f64::INFINITY,
                source_count: 0,
                timestamp: ts,
            };
        }

        let adjusted: Vec<f64> = active.iter().map(|s| s.value + s.bias).collect();
        let weights: Vec<f64> = active.iter().map(|s| s.confidence * s.weight).collect();
        let w_sum: f64 = weights.iter().sum();

        let value: f64 = adjusted
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum::<f64>()
            / w_sum;

        let mean = adjusted
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum::<f64>()
            / w_sum;
        let variance = adjusted
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| w * (v - mean).powi(2))
            .sum::<f64>()
            / w_sum;
        let confidence = weights.iter().sum::<f64>() / weights.len() as f64;
        let timestamp = active.iter().map(|s| s.timestamp).max().unwrap_or(0);

        FusedSignal {
            value,
            confidence,
            variance,
            source_count: active.len() as u8,
            timestamp,
        }
    }

    pub fn history(&self, n: usize) -> Vec<&FusedSignal> {
        self.history.iter().rev().take(n).collect()
    }

    pub fn agreement(&self) -> f64 {
        let active: Vec<&Sensor> = self
            .sensors
            .iter()
            .filter(|s| s.active && s.confidence > self.threshold)
            .collect();
        if active.len() < 2 {
            return 1.0;
        }
        let adjusted: Vec<f64> = active.iter().map(|s| s.value + s.bias).collect();
        let mean = adjusted.iter().sum::<f64>() / adjusted.len() as f64;
        let range = adjusted.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - adjusted.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        if range == 0.0 {
            return 1.0;
        }
        let norm_range = range / (mean.abs().max(1.0));
        1.0 / (1.0 + norm_range)
    }

    pub fn deactivate(&mut self, id: u8) {
        if let Some(sensor) = self.sensors.iter_mut().find(|s| s.id == id) {
            sensor.active = false;
        }
    }

    pub fn calibrate(&mut self, id: u8, bias: f64) {
        if let Some(sensor) = self.sensors.iter_mut().find(|s| s.id == id) {
            sensor.bias = bias;
        }
    }

    pub fn snapshot_and_record(&mut self) {
        let fused = self.read();
        self.history.push(fused);
    }
}

// ─── Sensor Data Pipeline ─────────────────────────────────────────────────

/// A raw data point from a sensor source.
#[derive(Clone, Debug)]
pub struct SensorReading {
    pub sensor_id: u8,
    pub value: f64,
    pub timestamp: u64,
}

/// Stage in the pipeline: ingest → filter → transform → emit.
#[derive(Clone, Debug, PartialEq)]
pub enum PipelineStage {
    Ingest,
    Filter,
    Transform,
    Emit,
}

/// A processed data point that has passed through pipeline stages.
#[derive(Clone, Debug)]
pub struct ProcessedReading {
    pub sensor_id: u8,
    pub value: f64,
    pub timestamp: u64,
    pub passed_filter: bool,
    pub transformed: bool,
}

/// Configuration for the data pipeline filters and transformations.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub min_value: f64,
    pub max_value: f64,
    pub scale_factor: f64,
    pub offset: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            min_value: f64::NEG_INFINITY,
            max_value: f64::INFINITY,
            scale_factor: 1.0,
            offset: 0.0,
        }
    }
}

/// Multi-stage sensor data pipeline.
pub struct SensorPipeline {
    config: PipelineConfig,
    buffer: Vec<ProcessedReading>,
    emitted: Vec<ProcessedReading>,
}

impl SensorPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            buffer: Vec::new(),
            emitted: Vec::new(),
        }
    }

    /// Ingest a raw reading into the pipeline.
    pub fn ingest(&mut self, reading: SensorReading) {
        let processed = ProcessedReading {
            sensor_id: reading.sensor_id,
            value: reading.value,
            timestamp: reading.timestamp,
            passed_filter: false,
            transformed: false,
        };
        self.buffer.push(processed);
    }

    /// Apply filter: keep only readings within [min_value, max_value].
    pub fn filter_stage(&mut self) -> usize {
        let mut passed = 0;
        for reading in &mut self.buffer {
            if reading.value >= self.config.min_value
                && reading.value <= self.config.max_value
            {
                reading.passed_filter = true;
                passed += 1;
            }
        }
        passed
    }

    /// Apply transform: value = value * scale_factor + offset (only on filtered readings).
    pub fn transform_stage(&mut self) -> usize {
        let mut transformed = 0;
        for reading in &mut self.buffer {
            if reading.passed_filter {
                reading.value = reading.value * self.config.scale_factor + self.config.offset;
                reading.transformed = true;
                transformed += 1;
            }
        }
        transformed
    }

    /// Emit all fully-processed readings (filtered + transformed).
    pub fn emit_stage(&mut self) -> Vec<ProcessedReading> {
        let ready: Vec<ProcessedReading> = self.buffer
            .iter()
            .filter(|r| r.passed_filter && r.transformed)
            .cloned()
            .collect();
        self.emitted.extend(ready.clone());
        ready
    }

    /// Run all stages in sequence: ingest already done, then filter → transform → emit.
    pub fn run(&mut self) -> Vec<ProcessedReading> {
        self.filter_stage();
        self.transform_stage();
        self.emit_stage()
    }

    /// Clear the internal buffer.
    pub fn clear_buffer(&mut self) {
        self.buffer.clear();
    }

    /// Get emitted readings count.
    pub fn emitted_count(&self) -> usize {
        self.emitted.len()
    }

    /// Get buffered readings count.
    pub fn buffered_count(&self) -> usize {
        self.buffer.len()
    }
}

// ─── Signal Processing ─────────────────────────────────────────────────────

/// Moving average filter over a window of values.
pub struct MovingAverage {
    window: Vec<f64>,
    capacity: usize,
}

impl MovingAverage {
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        Self {
            window: Vec::with_capacity(window_size),
            capacity: window_size,
        }
    }

    /// Push a new value and return the current moving average.
    pub fn push(&mut self, value: f64) -> f64 {
        if self.window.len() == self.capacity {
            self.window.remove(0);
        }
        self.window.push(value);
        self.window.iter().sum::<f64>() / self.window.len() as f64
    }

    /// Get the current average without pushing.
    pub fn current(&self) -> Option<f64> {
        if self.window.is_empty() {
            None
        } else {
            Some(self.window.iter().sum::<f64>() / self.window.len() as f64)
        }
    }

    /// Get window size.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
}

/// Result of peak detection.
#[derive(Clone, Debug, PartialEq)]
pub struct Peak {
    pub index: usize,
    pub value: f64,
}

/// Detect peaks in a signal. A peak is an element greater than both neighbors.
pub fn detect_peaks(signal: &[f64]) -> Vec<Peak> {
    let mut peaks = Vec::new();
    if signal.len() < 3 {
        return peaks;
    }
    for i in 1..signal.len() - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peaks.push(Peak { index: i, value: signal[i] });
        }
    }
    peaks
}

/// Threshold detection: returns indices where signal crosses above the threshold.
pub fn detect_threshold_crossings(signal: &[f64], threshold: f64) -> Vec<usize> {
    let mut crossings = Vec::new();
    for (i, &v) in signal.iter().enumerate() {
        if v > threshold {
            crossings.push(i);
        }
    }
    crossings
}

/// Zero-crossing detection for oscillating signals.
pub fn detect_zero_crossings(signal: &[f64]) -> Vec<usize> {
    let mut crossings = Vec::new();
    for i in 1..signal.len() {
        if (signal[i - 1] < 0.0 && signal[i] >= 0.0) || (signal[i - 1] >= 0.0 && signal[i] < 0.0) {
            crossings.push(i);
        }
    }
    crossings
}

// ─── Noise Filtering ───────────────────────────────────────────────────────

/// Median filter: replaces each value with the median of its neighborhood.
pub fn median_filter(signal: &[f64], window_size: usize) -> Vec<f64> {
    if signal.is_empty() || window_size == 0 {
        return signal.to_vec();
    }
    let half = window_size / 2;
    let mut result = Vec::with_capacity(signal.len());
    for i in 0..signal.len() {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(signal.len());
        let mut window: Vec<f64> = signal[start..end].to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = window.len() / 2;
        let median = if window.len() % 2 == 0 {
            (window[mid - 1] + window[mid]) / 2.0
        } else {
            window[mid]
        };
        result.push(median);
    }
    result
}

/// Simple exponential moving average (low-pass filter).
pub struct LowPassFilter {
    alpha: f64,
    output: Option<f64>,
}

impl LowPassFilter {
    pub fn new(alpha: f64) -> Self {
        assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0, 1]");
        Self { alpha, output: None }
    }

    /// Feed a new sample and get the filtered output.
    pub fn feed(&mut self, input: f64) -> f64 {
        self.output = Some(match self.output {
            None => input,
            Some(prev) => self.alpha * input + (1.0 - self.alpha) * prev,
        });
        self.output.unwrap()
    }

    /// Get current output without feeding new data.
    pub fn current(&self) -> Option<f64> {
        self.output
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.output = None;
    }
}

/// Remove outliers beyond `n_stddev` standard deviations from the mean.
pub fn remove_outliers(signal: &[f64], n_stddev: f64) -> Vec<f64> {
    if signal.len() < 2 {
        return signal.to_vec();
    }
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let variance = signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / signal.len() as f64;
    let stddev = variance.sqrt();
    signal.iter()
        .filter(|&&v| (v - mean).abs() <= n_stddev * stddev)
        .cloned()
        .collect()
}

// ─── Event Detection ───────────────────────────────────────────────────────

/// Types of detectable events in a signal stream.
#[derive(Clone, Debug, PartialEq)]
pub enum SignalEvent {
    /// Value crossed from below to above a threshold.
    RisingEdge { index: usize, value: f64, threshold: f64 },
    /// Value crossed from above to below a threshold.
    FallingEdge { index: usize, value: f64, threshold: f64 },
    /// Value changed level (crossed between low and high bands).
    LevelChange { index: usize, from_low: bool, to_low: bool, value: f64 },
    /// Value remained stable within a band for a duration.
    Stable { start: usize, end: usize, value: f64 },
}

/// Event detector that tracks previous values and emits events.
pub struct EventDetector {
    threshold: f64,
    hysteresis: f64,
    low_band: f64,
    high_band: f64,
    prev_value: Option<f64>,
    prev_above: Option<bool>,
}

impl EventDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            hysteresis: 0.0,
            low_band: threshold * 0.8,
            high_band: threshold * 1.2,
            prev_value: None,
            prev_above: None,
        }
    }

    pub fn with_bands(threshold: f64, low_band: f64, high_band: f64) -> Self {
        Self {
            threshold,
            hysteresis: 0.0,
            low_band,
            high_band,
            prev_value: None,
            prev_above: None,
        }
    }

    /// Process a single value and return any detected event.
    pub fn process(&mut self, value: f64) -> Option<SignalEvent> {
        let event = match self.prev_value {
            None => {
                let above = value > self.threshold;
                self.prev_above = Some(above);
                if above {
                    Some(SignalEvent::RisingEdge {
                        index: 0,
                        value,
                        threshold: self.threshold,
                    })
                } else {
                    None
                }
            }
            Some(prev) => {
                let was_above = self.prev_above.unwrap_or(false);
                let now_above = value > self.threshold;
                let mut event = None;

                if !was_above && now_above {
                    event = Some(SignalEvent::RisingEdge {
                        index: 0,
                        value,
                        threshold: self.threshold,
                    });
                } else if was_above && !now_above {
                    event = Some(SignalEvent::FallingEdge {
                        index: 0,
                        value,
                        threshold: self.threshold,
                    });
                }

                // Level change detection (only if no edge event)
                if event.is_none() {
                    let was_low = prev < self.low_band;
                    let now_low = value < self.low_band;
                    if was_low != now_low || (prev > self.high_band) != (value > self.high_band) {
                        event = Some(SignalEvent::LevelChange {
                            index: 0,
                            from_low: was_low,
                            to_low: now_low,
                            value,
                        });
                    }
                }

                self.prev_above = Some(now_above);
                event
            }
        };
        self.prev_value = Some(value);
        event
    }

    /// Process a full signal and return all detected events with proper indices.
    pub fn detect_all(&mut self, signal: &[f64]) -> Vec<SignalEvent> {
        self.reset();
        let mut events = Vec::new();
        for (i, &value) in signal.iter().enumerate() {
            if let Some(mut event) = self.process(value) {
                // Fix index
                match &mut event {
                    SignalEvent::RisingEdge { index, .. } => *index = i,
                    SignalEvent::FallingEdge { index, .. } => *index = i,
                    SignalEvent::LevelChange { index, .. } => *index = i,
                    _ => {}
                }
                events.push(event);
            }
        }
        events
    }

    /// Detect stable regions where the signal stays within a band.
    pub fn detect_stable(signal: &[f64], tolerance: f64, min_duration: usize) -> Vec<SignalEvent> {
        if signal.is_empty() {
            return Vec::new();
        }
        let mut events = Vec::new();
        let mut start = 0;
        let base = signal[0];

        for i in 1..signal.len() {
            if (signal[i] - base).abs() > tolerance {
                if i - start >= min_duration {
                    let avg = signal[start..i].iter().sum::<f64>() / (i - start) as f64;
                    events.push(SignalEvent::Stable {
                        start,
                        end: i - 1,
                        value: avg,
                    });
                }
                start = i;
            }
        }
        // Check trailing region
        if signal.len() - start >= min_duration {
            let avg = signal[start..].iter().sum::<f64>() / (signal.len() - start) as f64;
            events.push(SignalEvent::Stable {
                start,
                end: signal.len() - 1,
                value: avg,
            });
        }
        events
    }

    pub fn reset(&mut self) {
        self.prev_value = None;
        self.prev_above = None;
    }
}

// ─── Pattern Recognition (State Machine) ───────────────────────────────────

/// Recognizable signal patterns.
#[derive(Clone, Debug, PartialEq)]
pub enum Pattern {
    /// Gradual increase over time.
    TrendUp,
    /// Gradual decrease over time.
    TrendDown,
    /// Oscillating signal.
    Oscillation,
    /// Value staying within a narrow range.
    Stable,
    /// Spike followed by return to baseline.
    Spike,
    /// No recognizable pattern.
    Unknown,
}

/// State machine states for pattern recognition.
#[derive(Clone, Debug, PartialEq)]
enum PatternState {
    Initializing,
    Observing,
    TrendUp,
    TrendDown,
    Oscillation,
    Stable,
    Spike,
}

/// State machine based pattern recognizer.
pub struct PatternRecognizer {
    state: PatternState,
    history: Vec<f64>,
    window_size: usize,
    slope_threshold: f64,
    stability_threshold: f64,
}

impl PatternRecognizer {
    pub fn new(window_size: usize) -> Self {
        assert!(window_size >= 3, "window_size must be >= 3");
        Self {
            state: PatternState::Initializing,
            history: Vec::with_capacity(window_size),
            window_size,
            slope_threshold: 0.1,
            stability_threshold: 0.05,
        }
    }

    /// Feed a new value and get the current recognized pattern.
    pub fn feed(&mut self, value: f64) -> Pattern {
        self.history.push(value);
        if self.history.len() > self.window_size {
            self.history.remove(0);
        }

        if self.history.len() < self.window_size {
            return Pattern::Unknown;
        }

        let pattern = self.classify();
        self.state = match &pattern {
            Pattern::TrendUp => PatternState::TrendUp,
            Pattern::TrendDown => PatternState::TrendDown,
            Pattern::Oscillation => PatternState::Oscillation,
            Pattern::Stable => PatternState::Stable,
            Pattern::Spike => PatternState::Spike,
            Pattern::Unknown => PatternState::Observing,
        };
        pattern
    }

    fn classify(&self) -> Pattern {
        let h = &self.history;
        let n = h.len();

        // Check stability: all values within threshold
        let range = h.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - h.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        if range < self.stability_threshold {
            return Pattern::Stable;
        }

        // Calculate overall slope via linear regression
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean = h.iter().sum::<f64>() / n as f64;
        let numerator: f64 = (0..n).map(|i| (i as f64 - x_mean) * (h[i] - y_mean)).sum();
        let denominator: f64 = (0..n).map(|i| (i as f64 - x_mean).powi(2)).sum();
        let slope = if denominator.abs() < 1e-12 { 0.0 } else { numerator / denominator };

        // Check for trend before spike
        if slope > self.slope_threshold {
            return Pattern::TrendUp;
        }
        if slope < -self.slope_threshold {
            return Pattern::TrendDown;
        }

        // Check for spike: one value much larger than others (only when slope is flat)
        let mean = y_mean;
        let spike_count = h.iter().filter(|&&v| (v - mean).abs() > mean.abs() * 0.8 + 1.0).count();
        if spike_count == 1 && range > mean.abs().max(1.0) {
            return Pattern::Spike;
        }

        // Count direction changes for oscillation
        let mut dir_changes = 0;
        for i in 2..n {
            let d1 = h[i - 1] - h[i - 2];
            let d2 = h[i] - h[i - 1];
            if d1 * d2 < 0.0 {
                dir_changes += 1;
            }
        }
        if dir_changes >= 2 {
            return Pattern::Oscillation;
        }

        Pattern::Unknown
    }

    /// Classify a complete signal at once.
    pub fn classify_signal(signal: &[f64]) -> Pattern {
        let window = signal.len().max(3);
        let mut recognizer = PatternRecognizer::new(window);
        let mut last = Pattern::Unknown;
        for &v in signal {
            last = recognizer.feed(v);
        }
        last
    }

    /// Get current state.
    pub fn current_state(&self) -> &PatternState {
        &self.state
    }

    /// Reset the recognizer.
    pub fn reset(&mut self) {
        self.state = PatternState::Initializing;
        self.history.clear();
    }
}

// ─── Multi-Sensor Fusion ───────────────────────────────────────────────────

/// Fusion strategies for combining multiple sensor readings.
#[derive(Clone, Debug, PartialEq)]
pub enum FusionStrategy {
    /// Confidence-weighted average (default Engine behavior).
    WeightedAverage,
    /// Majority voting: pick the value closest to the median.
    Voting,
    /// Pick the sensor with highest confidence.
    WinnerTakeAll,
}

/// Enhanced multi-sensor fusion engine.
pub struct FusionEngine {
    sensors: Vec<Sensor>,
    strategy: FusionStrategy,
    confidence_threshold: f64,
}

impl FusionEngine {
    pub fn new(strategy: FusionStrategy, confidence_threshold: f64) -> Self {
        Self {
            sensors: Vec::new(),
            strategy,
            confidence_threshold,
        }
    }

    pub fn add_sensor(&mut self, id: u8, weight: f64, bias: f64) {
        self.sensors.push(Sensor {
            id,
            value: 0.0,
            confidence: 0.0,
            weight,
            bias,
            timestamp: 0,
            active: true,
        });
    }

    pub fn update(&mut self, sensor_id: u8, value: f64, confidence: f64, now: u64) {
        if let Some(sensor) = self.sensors.iter_mut().find(|s| s.id == sensor_id && s.active) {
            sensor.value = value;
            sensor.confidence = confidence;
            sensor.timestamp = now;
        }
    }

    /// Get active sensors above confidence threshold.
    fn active_sensors(&self) -> Vec<&Sensor> {
        self.sensors
            .iter()
            .filter(|s| s.active && s.confidence > self.confidence_threshold)
            .collect()
    }

    /// Fuse readings using the configured strategy.
    pub fn fuse(&self) -> FusedSignal {
        let active = self.active_sensors();

        if active.is_empty() {
            let ts = self.sensors.iter().map(|s| s.timestamp).max().unwrap_or(0);
            return FusedSignal { value: 0.0, confidence: 0.0, variance: f64::INFINITY, source_count: 0, timestamp: ts };
        }

        let adjusted: Vec<f64> = active.iter().map(|s| s.value + s.bias).collect();
        let confidences: Vec<f64> = active.iter().map(|s| s.confidence).collect();
        let timestamp = active.iter().map(|s| s.timestamp).max().unwrap_or(0);

        let (value, confidence, variance) = match self.strategy {
            FusionStrategy::WeightedAverage => {
                let weights: Vec<f64> = active.iter().map(|s| s.confidence * s.weight).collect();
                let w_sum: f64 = weights.iter().sum();
                let val = adjusted.iter().zip(weights.iter()).map(|(v, w)| v * w).sum::<f64>() / w_sum;
                let mean = val;
                let var = adjusted.iter().zip(weights.iter()).map(|(v, w)| w * (v - mean).powi(2)).sum::<f64>() / w_sum;
                let conf = confidences.iter().sum::<f64>() / confidences.len() as f64;
                (val, conf, var)
            }
            FusionStrategy::Voting => {
                // Median-based voting
                let mut sorted = adjusted.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let val = sorted[sorted.len() / 2];
                let mean = adjusted.iter().sum::<f64>() / adjusted.len() as f64;
                let var = adjusted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / adjusted.len() as f64;
                let conf = confidences.iter().sum::<f64>() / confidences.len() as f64;
                (val, conf, var)
            }
            FusionStrategy::WinnerTakeAll => {
                // Pick sensor with highest confidence
                let best = active.iter().max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap()).unwrap();
                let val = best.value + best.bias;
                (val, best.confidence, 0.0)
            }
        };

        FusedSignal {
            value,
            confidence,
            variance,
            source_count: active.len() as u8,
            timestamp,
        }
    }

    /// Set the fusion strategy.
    pub fn set_strategy(&mut self, strategy: FusionStrategy) {
        self.strategy = strategy;
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> Engine {
        let mut e = Engine::new(0.1);
        e.add_sensor(1, 1.0, 0.0);
        e.add_sensor(2, 1.0, 0.0);
        e.add_sensor(3, 0.5, 0.0);
        e
    }

    // ── Original Engine tests ──

    #[test]
    fn test_new_engine_empty() {
        let e = Engine::new(0.1);
        assert!(e.sensors.is_empty());
        assert!(e.history.is_empty());
    }

    #[test]
    fn test_add_sensor() {
        let e = make_engine();
        assert_eq!(e.sensors.len(), 3);
        assert_eq!(e.sensors[0].id, 1);
        assert_eq!(e.sensors[0].weight, 1.0);
    }

    #[test]
    fn test_find_sensor_exists() {
        let e = make_engine();
        assert!(e.find_sensor(2).is_some());
        assert_eq!(e.find_sensor(2).unwrap().id, 2);
    }

    #[test]
    fn test_find_sensor_missing() {
        let e = make_engine();
        assert!(e.find_sensor(99).is_none());
    }

    #[test]
    fn test_update_sensor() {
        let mut e = make_engine();
        e.update(1, 42.0, 0.9, 100);
        let s = e.find_sensor(1).unwrap();
        assert_eq!(s.value, 42.0);
        assert_eq!(s.confidence, 0.9);
        assert_eq!(s.timestamp, 100);
    }

    #[test]
    fn test_update_inactive_ignored() {
        let mut e = make_engine();
        e.deactivate(1);
        e.update(1, 42.0, 0.9, 100);
        let s = e.find_sensor(1).unwrap();
        assert_eq!(s.value, 0.0);
    }

    #[test]
    fn test_update_nonexistent_ignored() {
        let mut e = make_engine();
        e.update(99, 1.0, 1.0, 0);
        assert_eq!(e.sensors.len(), 3);
    }

    #[test]
    fn test_read_weighted_average() {
        let mut e = make_engine();
        e.update(1, 10.0, 1.0, 1);
        e.update(2, 20.0, 1.0, 1);
        let f = e.read();
        assert_eq!(f.value, 15.0);
        assert_eq!(f.source_count, 2);
    }

    #[test]
    fn test_read_confidence_weighted() {
        let mut e = make_engine();
        e.update(1, 10.0, 0.8, 1);
        e.update(2, 20.0, 0.2, 1);
        let f = e.read();
        assert!((f.value - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_read_with_bias() {
        let mut e = Engine::new(0.1);
        e.add_sensor(1, 1.0, 5.0);
        e.add_sensor(2, 1.0, 0.0);
        e.update(1, 10.0, 1.0, 1);
        e.update(2, 20.0, 1.0, 1);
        let f = e.read();
        assert_eq!(f.value, 17.5);
    }

    #[test]
    fn test_read_empty_returns_zero() {
        let e = Engine::new(0.1);
        let f = e.read();
        assert_eq!(f.value, 0.0);
        assert_eq!(f.source_count, 0);
    }

    #[test]
    fn test_read_below_threshold_excluded() {
        let mut e = make_engine();
        e.update(1, 10.0, 0.05, 1);
        let f = e.read();
        assert_eq!(f.source_count, 0);
    }

    #[test]
    fn test_agreement_perfect() {
        let mut e = make_engine();
        e.update(1, 10.0, 0.9, 1);
        e.update(2, 10.0, 0.9, 1);
        assert_eq!(e.agreement(), 1.0);
    }

    #[test]
    fn test_agreement_low() {
        let mut e = make_engine();
        e.update(1, 0.0, 0.9, 1);
        e.update(2, 100.0, 0.9, 1);
        assert!(e.agreement() < 0.5);
    }

    #[test]
    fn test_agreement_single_sensor() {
        let mut e = make_engine();
        e.update(1, 42.0, 0.9, 1);
        assert_eq!(e.agreement(), 1.0);
    }

    #[test]
    fn test_deactivate() {
        let mut e = make_engine();
        e.deactivate(2);
        assert!(!e.find_sensor(2).unwrap().active);
        assert!(e.find_sensor(1).unwrap().active);
    }

    #[test]
    fn test_calibrate() {
        let mut e = make_engine();
        e.calibrate(1, 3.14);
        assert!((e.find_sensor(1).unwrap().bias - 3.14).abs() < 1e-9);
    }

    #[test]
    fn test_history_returns_last_n() {
        let mut e = make_engine();
        e.history.push(FusedSignal {
            value: 1.0,
            confidence: 0.5,
            variance: 0.1,
            source_count: 1,
            timestamp: 1,
        });
        e.history.push(FusedSignal {
            value: 2.0,
            confidence: 0.6,
            variance: 0.2,
            source_count: 2,
            timestamp: 2,
        });
        e.history.push(FusedSignal {
            value: 3.0,
            confidence: 0.7,
            variance: 0.3,
            source_count: 3,
            timestamp: 3,
        });
        let h = e.history(2);
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].value, 3.0);
        assert_eq!(h[1].value, 2.0);
    }

    #[test]
    fn test_variance_calculation() {
        let mut e = make_engine();
        e.update(1, 0.0, 1.0, 1);
        e.update(2, 10.0, 1.0, 1);
        let f = e.read();
        assert!((f.variance - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_weight_affects_value() {
        let mut e = Engine::new(0.1);
        e.add_sensor(1, 3.0, 0.0);
        e.add_sensor(2, 1.0, 0.0);
        e.update(1, 10.0, 1.0, 1);
        e.update(2, 22.0, 1.0, 1);
        let f = e.read();
        assert!((f.value - 13.0).abs() < 1e-9);
    }

    // ── Sensor Data Pipeline tests ──

    #[test]
    fn test_pipeline_ingest_and_buffer() {
        let config = PipelineConfig::default();
        let mut pipeline = SensorPipeline::new(config);
        pipeline.ingest(SensorReading { sensor_id: 1, value: 10.0, timestamp: 1 });
        pipeline.ingest(SensorReading { sensor_id: 2, value: 20.0, timestamp: 2 });
        assert_eq!(pipeline.buffered_count(), 2);
    }

    #[test]
    fn test_pipeline_filter_passes_valid_range() {
        let config = PipelineConfig { min_value: 5.0, max_value: 15.0, scale_factor: 1.0, offset: 0.0 };
        let mut pipeline = SensorPipeline::new(config);
        pipeline.ingest(SensorReading { sensor_id: 1, value: 10.0, timestamp: 1 });
        pipeline.ingest(SensorReading { sensor_id: 2, value: 3.0, timestamp: 2 });
        pipeline.ingest(SensorReading { sensor_id: 3, value: 20.0, timestamp: 3 });
        let passed = pipeline.filter_stage();
        assert_eq!(passed, 1); // only 10.0 is in range
    }

    #[test]
    fn test_pipeline_transform_applies_scale_and_offset() {
        let config = PipelineConfig { min_value: 0.0, max_value: 100.0, scale_factor: 2.0, offset: 5.0 };
        let mut pipeline = SensorPipeline::new(config);
        pipeline.ingest(SensorReading { sensor_id: 1, value: 10.0, timestamp: 1 });
        pipeline.run();
        let emitted = pipeline.emit_stage();
        assert_eq!(emitted.len(), 1);
        assert!((emitted[0].value - 25.0).abs() < 1e-9); // 10 * 2 + 5
    }

    #[test]
    fn test_pipeline_full_run() {
        let config = PipelineConfig { min_value: 0.0, max_value: 50.0, scale_factor: 1.0, offset: 0.0 };
        let mut pipeline = SensorPipeline::new(config);
        pipeline.ingest(SensorReading { sensor_id: 1, value: 10.0, timestamp: 1 });
        pipeline.ingest(SensorReading { sensor_id: 2, value: 60.0, timestamp: 2 }); // filtered out
        pipeline.ingest(SensorReading { sensor_id: 3, value: 30.0, timestamp: 3 });
        let emitted = pipeline.run();
        assert_eq!(emitted.len(), 2);
    }

    #[test]
    fn test_pipeline_clear_buffer() {
        let config = PipelineConfig::default();
        let mut pipeline = SensorPipeline::new(config);
        pipeline.ingest(SensorReading { sensor_id: 1, value: 1.0, timestamp: 1 });
        pipeline.clear_buffer();
        assert_eq!(pipeline.buffered_count(), 0);
    }

    // ── Signal Processing tests ──

    #[test]
    fn test_moving_average_basic() {
        let mut ma = MovingAverage::new(3);
        assert_eq!(ma.push(10.0), 10.0);
        assert_eq!(ma.push(20.0), 15.0);
        assert_eq!(ma.push(30.0), 20.0);
        assert_eq!(ma.push(40.0), 30.0); // drops 10
    }

    #[test]
    fn test_moving_average_window_capacity() {
        let mut ma = MovingAverage::new(2);
        ma.push(10.0);
        ma.push(20.0);
        ma.push(30.0);
        assert_eq!(ma.len(), 2);
        assert_eq!(ma.current().unwrap(), 25.0);
    }

    #[test]
    fn test_moving_average_empty() {
        let ma = MovingAverage::new(3);
        assert!(ma.current().is_none());
        assert!(ma.is_empty());
    }

    #[test]
    fn test_detect_peaks_basic() {
        let signal = vec![1.0, 3.0, 2.0, 5.0, 1.0, 4.0, 2.0];
        let peaks = detect_peaks(&signal);
        assert_eq!(peaks.len(), 3);
        assert_eq!(peaks[0], Peak { index: 1, value: 3.0 });
        assert_eq!(peaks[1], Peak { index: 3, value: 5.0 });
        assert_eq!(peaks[2], Peak { index: 5, value: 4.0 });
    }

    #[test]
    fn test_detect_peaks_short_signal() {
        assert!(detect_peaks(&[1.0, 2.0]).is_empty());
        assert!(detect_peaks(&[1.0]).is_empty());
    }

    #[test]
    fn test_detect_threshold_crossings() {
        let signal = vec![1.0, 2.0, 5.0, 3.0, 0.5, 6.0];
        let crossings = detect_threshold_crossings(&signal, 3.0);
        assert_eq!(crossings, vec![2, 5]);
    }

    #[test]
    fn test_detect_zero_crossings() {
        let signal = vec![-1.0, 1.0, -1.0, 1.0];
        let crossings = detect_zero_crossings(&signal);
        assert_eq!(crossings.len(), 3);
    }

    // ── Noise Filtering tests ──

    #[test]
    fn test_median_filter_removes_spike() {
        let signal = vec![1.0, 2.0, 100.0, 2.0, 1.0];
        let filtered = median_filter(&signal, 3);
        assert!(filtered[2] < 100.0); // spike should be reduced
    }

    #[test]
    fn test_median_filter_preserves_smooth() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let filtered = median_filter(&signal, 3);
        assert_eq!(filtered[0], 1.5); // median of [1, 2]
        assert_eq!(filtered[2], 3.0); // median of [2, 3, 4]
    }

    #[test]
    fn test_low_pass_filter_smoothing() {
        let mut lpf = LowPassFilter::new(0.5);
        let out1 = lpf.feed(10.0);
        assert_eq!(out1, 10.0); // first value passes through
        let out2 = lpf.feed(20.0);
        // 0.5 * 20 + 0.5 * 10 = 15
        assert!((out2 - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_low_pass_filter_reset() {
        let mut lpf = LowPassFilter::new(0.5);
        lpf.feed(10.0);
        lpf.reset();
        assert!(lpf.current().is_none());
        assert_eq!(lpf.feed(20.0), 20.0); // fresh start
    }

    #[test]
    fn test_remove_outliers() {
        let signal = vec![10.0, 11.0, 10.5, 100.0, 10.2];
        let cleaned = remove_outliers(&signal, 1.0);
        assert!(!cleaned.contains(&100.0));
        assert_eq!(cleaned.len(), 4);
    }

    #[test]
    fn test_remove_outliers_short_signal() {
        let signal = vec![42.0];
        let cleaned = remove_outliers(&signal, 2.0);
        assert_eq!(cleaned.len(), 1);
    }

    // ── Event Detection tests ──

    #[test]
    fn test_event_rising_edge() {
        let mut detector = EventDetector::new(5.0);
        detector.process(1.0);
        let event = detector.process(10.0);
        assert!(matches!(event, Some(SignalEvent::RisingEdge { .. })));
    }

    #[test]
    fn test_event_falling_edge() {
        let mut detector = EventDetector::new(5.0);
        detector.process(10.0);
        let event = detector.process(1.0);
        assert!(matches!(event, Some(SignalEvent::FallingEdge { .. })));
    }

    #[test]
    fn test_event_no_edge_same_side() {
        let mut detector = EventDetector::new(5.0);
        detector.process(1.0);
        let event = detector.process(3.0);
        assert!(event.is_none()); // still below threshold
    }

    #[test]
    fn test_detect_stable_region() {
        let signal = vec![10.0, 10.1, 10.05, 9.95, 20.0, 30.0, 10.0, 10.0, 10.0];
        let stable = EventDetector::detect_stable(&signal, 0.2, 3);
        assert!(!stable.is_empty());
        assert!(matches!(stable[0], SignalEvent::Stable { start: 0, .. }));
    }

    #[test]
    fn test_detect_all_events_on_signal() {
        let signal = vec![1.0, 10.0, 1.0, 10.0];
        let mut detector = EventDetector::new(5.0);
        let events = detector.detect_all(&signal);
        assert!(!events.is_empty());
        // Should detect alternating rising and falling edges
        let rising_count = events.iter().filter(|e| matches!(e, SignalEvent::RisingEdge { .. })).count();
        let falling_count = events.iter().filter(|e| matches!(e, SignalEvent::FallingEdge { .. })).count();
        assert_eq!(rising_count, 2);
        assert_eq!(falling_count, 1);
    }

    #[test]
    fn test_event_detector_with_custom_bands() {
        let detector = EventDetector::with_bands(5.0, 2.0, 8.0);
        assert!((detector.low_band - 2.0).abs() < 1e-9);
        assert!((detector.high_band - 8.0).abs() < 1e-9);
    }

    // ── Pattern Recognition tests ──

    #[test]
    fn test_pattern_stable() {
        let mut pr = PatternRecognizer::new(5);
        let pattern = pr.feed(10.0);
        assert_eq!(pattern, Pattern::Unknown); // not enough data yet
        for _ in 0..4 {
            pr.feed(10.01);
        }
        assert_eq!(pr.feed(10.0), Pattern::Stable);
    }

    #[test]
    fn test_pattern_trend_up() {
        let signal = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let pattern = PatternRecognizer::classify_signal(&signal);
        assert_eq!(pattern, Pattern::TrendUp);
    }

    #[test]
    fn test_pattern_trend_down() {
        let signal = vec![16.0, 8.0, 4.0, 2.0, 1.0];
        let pattern = PatternRecognizer::classify_signal(&signal);
        assert_eq!(pattern, Pattern::TrendDown);
    }

    #[test]
    fn test_pattern_oscillation() {
        let signal = vec![1.0, 10.0, 1.0, 10.0, 1.0];
        let pattern = PatternRecognizer::classify_signal(&signal);
        assert_eq!(pattern, Pattern::Oscillation);
    }

    #[test]
    fn test_pattern_spike() {
        let signal = vec![10.0, 10.0, 100.0, 10.0, 10.0];
        let pattern = PatternRecognizer::classify_signal(&signal);
        assert_eq!(pattern, Pattern::Spike);
    }

    #[test]
    fn test_pattern_recognizer_reset() {
        let mut pr = PatternRecognizer::new(3);
        pr.feed(1.0);
        pr.feed(2.0);
        pr.reset();
        assert!(pr.history.is_empty());
        assert_eq!(pr.current_state(), &PatternState::Initializing);
    }

    // ── Multi-Sensor Fusion tests ──

    #[test]
    fn test_fusion_weighted_average() {
        let mut fe = FusionEngine::new(FusionStrategy::WeightedAverage, 0.1);
        fe.add_sensor(1, 1.0, 0.0);
        fe.add_sensor(2, 1.0, 0.0);
        fe.update(1, 10.0, 0.8, 1);
        fe.update(2, 20.0, 0.2, 1);
        let f = fe.fuse();
        assert!((f.value - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_fusion_voting_median() {
        let mut fe = FusionEngine::new(FusionStrategy::Voting, 0.1);
        fe.add_sensor(1, 1.0, 0.0);
        fe.add_sensor(2, 1.0, 0.0);
        fe.add_sensor(3, 1.0, 0.0);
        fe.update(1, 10.0, 1.0, 1);
        fe.update(2, 20.0, 1.0, 1);
        fe.update(3, 30.0, 1.0, 1);
        let f = fe.fuse();
        assert_eq!(f.value, 20.0); // median of [10, 20, 30]
    }

    #[test]
    fn test_fusion_winner_take_all() {
        let mut fe = FusionEngine::new(FusionStrategy::WinnerTakeAll, 0.1);
        fe.add_sensor(1, 1.0, 0.0);
        fe.add_sensor(2, 1.0, 0.0);
        fe.update(1, 10.0, 0.3, 1);
        fe.update(2, 20.0, 0.9, 1);
        let f = fe.fuse();
        assert_eq!(f.value, 20.0);
        assert!((f.confidence - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_fusion_empty_returns_zero() {
        let fe = FusionEngine::new(FusionStrategy::WeightedAverage, 0.1);
        let f = fe.fuse();
        assert_eq!(f.source_count, 0);
        assert_eq!(f.value, 0.0);
    }

    #[test]
    fn test_fusion_set_strategy() {
        let mut fe = FusionEngine::new(FusionStrategy::WeightedAverage, 0.1);
        fe.set_strategy(FusionStrategy::Voting);
        assert_eq!(fe.strategy, FusionStrategy::Voting);
    }

    #[test]
    fn test_fusion_below_threshold_excluded() {
        let mut fe = FusionEngine::new(FusionStrategy::WeightedAverage, 0.5);
        fe.add_sensor(1, 1.0, 0.0);
        fe.update(1, 10.0, 0.1, 1); // below 0.5
        let f = fe.fuse();
        assert_eq!(f.source_count, 0);
    }

    // ── Snapshot and record test ──

    #[test]
    fn test_snapshot_and_record() {
        let mut e = make_engine();
        e.update(1, 10.0, 0.9, 1);
        e.snapshot_and_record();
        assert_eq!(e.history.len(), 1);
        let h = e.history(1);
        assert!((h[0].value - 10.0).abs() < 1e-9);
    }
}
