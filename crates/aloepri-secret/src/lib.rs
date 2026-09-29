//! Secret output is intentionally absent in v0.1.

use aloepri_core::{
    error::{CompilerError, Result},
    plan::TransformConfig,
};

pub fn validate_v0_1_boundary(config: &TransformConfig) -> Result<()> {
    if config.method.id != "identity" {
        return Err(CompilerError::Unsupported(
            "Client Secret and mathematical transforms are not implemented in v0.1".into(),
        ));
    }
    Ok(())
}

pub fn secret_output_is_absent() -> Option<&'static str> {
    None
}
