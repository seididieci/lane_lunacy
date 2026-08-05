// SPDX-License-Identifier: MIT

/// Sky/weather state driving cloud coverage. `Auto` runs a slow dynamic cycle
/// with a per-run random start; the fixed states override it. `Rain` currently
/// only means full overcast clouds — task 4 wires actual rain particles to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weather {
    Auto,
    Clear,
    Cloudy,
    Rain,
}

impl Weather {
    pub fn label(self) -> &'static str {
        match self {
            Weather::Auto => "AUTO",
            Weather::Clear => "CLEAR",
            Weather::Cloudy => "CLOUDY",
            Weather::Rain => "RAIN",
        }
    }

    /// Parses a `--weather` CLI value (case-insensitive) into a weather state.
    pub fn parse(s: &str) -> Option<Weather> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Weather::Auto),
            "clear" => Some(Weather::Clear),
            "cloudy" => Some(Weather::Cloudy),
            "rain" => Some(Weather::Rain),
            _ => None,
        }
    }

    /// Base cloud coverage (0..1) for the fixed states, spread across the full
    /// range so each weather setting reads as a clearly different sky.
    pub fn cloud_amount(self) -> f32 {
        match self {
            Weather::Auto => 0.5,
            Weather::Clear => 0.15,
            Weather::Cloudy => 0.65,
            Weather::Rain => 1.0,
        }
    }
}
