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
}

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
        e.update(1, 10.0, 0.05, 1); // below 0.1 threshold
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
        // weighted: (10*3 + 22*1)/(3+1) = 52/4 = 13
        assert!((f.value - 13.0).abs() < 1e-9);
    }

    #[test]
    fn test_deactivate_nonexistent() {
        let mut e = make_engine();
        e.deactivate(99); // should not panic
    }

    #[test]
    fn test_calibrate_nonexistent() {
        let mut e = make_engine();
        e.calibrate(99, 3.14); // should not panic
    }

    #[test]
    fn test_agreement_with_deactivated() {
        let mut e = make_engine();
        e.update(1, 10.0, 0.9, 1);
        e.update(2, 10.0, 0.9, 1);
        e.deactivate(2);
        assert_eq!(e.agreement(), 1.0); // single active -> always 1.0
    }

    #[test]
    fn test_read_mixed_active_inactive() {
        let mut e = make_engine();
        e.update(1, 10.0, 0.9, 1);
        e.update(2, 20.0, 0.9, 1);
        e.deactivate(2);
        let f = e.read();
        assert_eq!(f.value, 10.0);
        assert_eq!(f.source_count, 1);
    }

    #[test]
    fn test_history_empty() {
        let e = make_engine();
        let h = e.history(5);
        assert!(h.is_empty());
    }

    #[test]
    fn test_history_request_more_than_available() {
        let mut e = make_engine();
        e.history.push(FusedSignal {
            value: 1.0,
            confidence: 0.5,
            variance: 0.1,
            source_count: 1,
            timestamp: 1,
        });
        let h = e.history(10);
        assert_eq!(h.len(), 1);
    }

    #[test]
    fn test_variance_zero_when_all_same() {
        let mut e = make_engine();
        e.update(1, 10.0, 1.0, 1);
        e.update(2, 10.0, 1.0, 1);
        let f = e.read();
        assert!((f.variance - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_read_timestamp_latest() {
        let mut e = make_engine();
        e.update(1, 10.0, 0.9, 100);
        e.update(2, 20.0, 0.9, 200);
        let f = e.read();
        assert_eq!(f.timestamp, 200);
    }

    #[test]
    fn test_sensor_default_values() {
        let e = make_engine();
        let s = e.find_sensor(1).unwrap();
        assert_eq!(s.value, 0.0);
        assert_eq!(s.confidence, 0.0);
        assert_eq!(s.timestamp, 0);
        assert!(s.active);
    }

    #[test]
    fn test_agreement_all_deactivated() {
        let mut e = make_engine();
        e.deactivate(1);
        e.deactivate(2);
        e.deactivate(3);
        assert_eq!(e.agreement(), 1.0); // no active sensors -> 1.0
    }

    #[test]
    fn test_read_empty_variance_infinity() {
        let e = Engine::new(0.1);
        let f = e.read();
        assert!(f.variance.is_infinite());
    }

    #[test]
    fn test_confidence_field_in_read() {
        let mut e = make_engine();
        e.update(1, 10.0, 1.0, 1);
        e.update(2, 20.0, 0.5, 1);
        let f = e.read();
        // confidence = avg of (conf*weight): (1.0*1.0 + 0.5*1.0)/2 = 0.75
        assert!((f.confidence - 0.75).abs() < 1e-9);
    }
}
