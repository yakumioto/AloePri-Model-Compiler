use crate::error::{CompilerError, Result};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Clone, Debug)]
pub struct MemoryBudget {
    limit: u64,
    used: Arc<AtomicU64>,
}

impl MemoryBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            used: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Acquire)
    }

    pub fn available(&self) -> u64 {
        self.limit.saturating_sub(self.used())
    }

    pub fn reserve(&self, bytes: u64) -> Result<MemoryReservation> {
        loop {
            let current = self.used.load(Ordering::Acquire);
            let next = current
                .checked_add(bytes)
                .ok_or(CompilerError::MemoryLimitExceeded {
                    requested: bytes,
                    available: self.available(),
                })?;
            if next > self.limit {
                return Err(CompilerError::MemoryLimitExceeded {
                    requested: bytes,
                    available: self.limit.saturating_sub(current),
                });
            }
            if self
                .used
                .compare_exchange(current, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(MemoryReservation {
                    used: Arc::clone(&self.used),
                    bytes,
                });
            }
        }
    }
}

#[derive(Debug)]
pub struct MemoryReservation {
    used: Arc<AtomicU64>,
    bytes: u64,
}

impl MemoryReservation {
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_release_on_drop() {
        let budget = MemoryBudget::new(10);
        let reservation = budget.reserve(7).unwrap();
        assert_eq!(budget.used(), 7);
        drop(reservation);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn budget_rejects_over_limit() {
        let budget = MemoryBudget::new(10);
        assert!(budget.reserve(11).is_err());
    }
}
