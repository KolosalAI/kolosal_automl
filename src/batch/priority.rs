//! Priority Queue Implementation
//!
//! Provides priority-based ordering for batch processing requests.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::Instant;

/// Priority levels for batch processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Priority {
    /// Highest priority - process immediately
    Critical = 0,
    /// High priority
    High = 1,
    /// Normal priority (default)
    Normal = 2,
    /// Low priority
    Low = 3,
    /// Background priority - process when idle
    Background = 4,
}

impl Default for Priority {
    fn default() -> Self {
        Priority::Normal
    }
}

impl From<u8> for Priority {
    fn from(value: u8) -> Self {
        match value {
            0 => Priority::Critical,
            1 => Priority::High,
            2 => Priority::Normal,
            3 => Priority::Low,
            _ => Priority::Background,
        }
    }
}

/// An item with priority for queue ordering
#[derive(Debug, Clone)]
pub struct PrioritizedItem<T> {
    /// Priority level (lower = higher priority)
    pub priority: Priority,
    /// Timestamp when item was added (for FIFO within same priority)
    pub timestamp: Instant,
    /// The actual item
    pub item: T,
    /// Unique sequence number for stable ordering
    pub sequence: u64,
}

impl<T> PrioritizedItem<T> {
    /// Create a new prioritized item
    pub fn new(item: T, priority: Priority, sequence: u64) -> Self {
        Self {
            priority,
            timestamp: Instant::now(),
            item,
            sequence,
        }
    }
    
    /// Create with normal priority
    pub fn normal(item: T, sequence: u64) -> Self {
        Self::new(item, Priority::Normal, sequence)
    }
    
    /// Create with high priority
    pub fn high(item: T, sequence: u64) -> Self {
        Self::new(item, Priority::High, sequence)
    }
    
    /// Create with critical priority
    pub fn critical(item: T, sequence: u64) -> Self {
        Self::new(item, Priority::Critical, sequence)
    }
}

impl<T> PartialEq for PrioritizedItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl<T> Eq for PrioritizedItem<T> {}

impl<T> PartialOrd for PrioritizedItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for PrioritizedItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower priority value = higher priority (should come first)
        // For max-heap, we reverse the comparison
        match (self.priority as u8).cmp(&(other.priority as u8)) {
            Ordering::Equal => {
                // Within same priority, use sequence (FIFO - earlier first)
                // Reverse for max-heap
                other.sequence.cmp(&self.sequence)
            }
            other_ord => other_ord.reverse(),
        }
    }
}

/// Thread-safe priority queue for batch processing
pub struct PriorityQueue<T> {
    heap: BinaryHeap<PrioritizedItem<T>>,
    sequence_counter: u64,
    max_size: usize,
}

impl<T> PriorityQueue<T> {
    /// Create a new priority queue with given maximum size
    pub fn new(max_size: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(max_size.min(1024)),
            sequence_counter: 0,
            max_size,
        }
    }
    
    /// Push an item with the given priority
    pub fn push(&mut self, item: T, priority: Priority) -> bool {
        if self.heap.len() >= self.max_size {
            return false;
        }
        
        let prioritized = PrioritizedItem::new(item, priority, self.sequence_counter);
        self.sequence_counter = self.sequence_counter.wrapping_add(1);
        self.heap.push(prioritized);
        true
    }
    
    /// Push an item with normal priority
    pub fn push_normal(&mut self, item: T) -> bool {
        self.push(item, Priority::Normal)
    }
    
    /// Pop the highest priority item
    pub fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|p| p.item)
    }
    
    /// Pop with priority information
    pub fn pop_with_priority(&mut self) -> Option<PrioritizedItem<T>> {
        self.heap.pop()
    }
    
    /// Peek at the highest priority item
    pub fn peek(&self) -> Option<&T> {
        self.heap.peek().map(|p| &p.item)
    }
    
    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
    
    /// Get the current queue length
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    
    /// Check if the queue is full
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.max_size
    }
    
    /// Get the maximum size
    pub fn capacity(&self) -> usize {
        self.max_size
    }
    
    /// Clear all items from the queue
    pub fn clear(&mut self) {
        self.heap.clear();
    }
    
    /// Drain up to `n` items from the queue
    pub fn drain(&mut self, n: usize) -> Vec<T> {
        let mut items = Vec::with_capacity(n.min(self.heap.len()));
        for _ in 0..n {
            match self.heap.pop() {
                Some(p) => items.push(p.item),
                None => break,
            }
        }
        items
    }
    
    /// Drain all items from the queue
    pub fn drain_all(&mut self) -> Vec<T> {
        let items: Vec<T> = std::mem::take(&mut self.heap)
            .into_sorted_vec()
            .into_iter()
            .map(|p| p.item)
            .collect();
        items
    }
}

impl<T> Default for PriorityQueue<T> {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_priority_ordering() {
        let mut queue: PriorityQueue<&str> = PriorityQueue::new(100);
        
        queue.push("low", Priority::Low);
        queue.push("critical", Priority::Critical);
        queue.push("normal", Priority::Normal);
        queue.push("high", Priority::High);
        
        assert_eq!(queue.pop(), Some("critical"));
        assert_eq!(queue.pop(), Some("high"));
        assert_eq!(queue.pop(), Some("normal"));
        assert_eq!(queue.pop(), Some("low"));
    }
    
    #[test]
    fn test_fifo_within_priority() {
        let mut queue: PriorityQueue<i32> = PriorityQueue::new(100);
        
        queue.push(1, Priority::Normal);
        queue.push(2, Priority::Normal);
        queue.push(3, Priority::Normal);
        
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
    }
    
    #[test]
    fn test_max_size() {
        let mut queue: PriorityQueue<i32> = PriorityQueue::new(3);
        
        assert!(queue.push(1, Priority::Normal));
        assert!(queue.push(2, Priority::Normal));
        assert!(queue.push(3, Priority::Normal));
        assert!(!queue.push(4, Priority::Normal)); // Should fail
        
        assert!(queue.is_full());
    }
}
