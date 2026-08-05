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

    /// Base cloud coverage (0..1) for the fixed states; `Auto` resolves to a
    /// value mid-range that the run cycle animates around.
    pub fn cloud_amount(self) -> f32 {
        match self {
            Weather::Auto => 0.5,
            Weather::Clear => 0.15,
            Weather::Cloudy => 0.55,
            Weather::Rain => 0.9,
        }
    }
}
