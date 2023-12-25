//! Defines a polymorphic allocator type. This crate provides the type `Polymorphic<'alloc>`, a uniform representation
//! for an allocator of any type, like `dyn Allocator + 'alloc`, but which can be freely cloned and stored. Under the hood,
//! the backing allocator is stored in a reference-counted allocation.

#![no_std]
#![feature(allocator_api)]
#![forbid(unsafe_op_in_unsafe_fn)]

use core::{
    alloc::{AllocError, Allocator, Layout},
    ptr::{self, NonNull},
    sync::atomic::{self, AtomicUsize, Ordering},
};

struct PolymorphicInner<A: Allocator + ?Sized> {
    ref_count: AtomicUsize,
    dtor: unsafe fn(*const ()),
    alloc: A,
}

/// An allocator that can wrap any (`Sized`) allocator. See the crate documentation for details.
#[derive(Debug, PartialEq, Eq)]
pub struct Polymorphic<'alloc> {
    /// The allocator being wrapped, reference-counted. Safety invariant: this pointer is valid, and the allocator
    /// that allocated the storage is compatible with the stored allocator (i.e. the allocator is stored in its own
    /// allocation).
    value: NonNull<PolymorphicInner<dyn Allocator + Send + Sync + 'alloc>>,
}

impl<'alloc> Polymorphic<'alloc> {
    /// Constructs a new `Polymorphic` wrapping the given allocator. The allocator must be thread-safe.
    ///
    /// # Errors
    /// This function returns an error if storage for the allocator cannot be allocated.
    pub fn try_new<A>(alloc: A) -> Result<Self, A>
    where
        A: Allocator + Send + Sync + 'alloc,
    {
        /// # Safety
        /// The storage must hold an value of type `PolymorphicInner<A>`, and must also have been allocated by the allocator
        /// inside.
        unsafe fn drop_alloc<A: Allocator>(storage: *const ()) {
            let storage: *const PolymorphicInner<A> = storage.cast::<PolymorphicInner<A>>();
            // SAFETY: This is a shared allocation, so we're allowed to take by shared reference.
            let ref_count = unsafe { &(*storage).ref_count };
            // decrement the ref-count, and do deallocation if we are the final owner. See std::sync::Arc for details
            // on why this works the way it does
            if ref_count.fetch_sub(1, Ordering::Release) == 1 {
                // sequence the deallocation after all other owners' dtors
                atomic::fence(Ordering::Acquire);
                // SAFETY: We have exclusive access to the inner value, so we can mutate / drop it.
                // Further, the caller has guaranteed that the storage was allocated by the allocator.
                unsafe {
                    let storage = storage.cast_mut();
                    let alloc = ptr::addr_of_mut!((*storage).alloc).read();
                    let layout = Layout::new::<PolymorphicInner<A>>();
                    alloc.deallocate(NonNull::new_unchecked(storage.cast()), layout)
                }
            }
        }

        let layout = Layout::new::<PolymorphicInner<A>>();
        let Ok(storage) = alloc.allocate(layout) else {
            return Err(alloc);
        };
        let storage = storage.cast::<PolymorphicInner<A>>();
        // SAFETY: we store the allocator inside its own ref-counted allocation, satisfying the type invariants
        unsafe {
            storage.as_ptr().write(PolymorphicInner {
                ref_count: AtomicUsize::new(1),
                dtor: drop_alloc::<A>,
                alloc,
            });
            Ok(Self { value: storage })
        }
    }

    /// Constructs a new `Polymorphic` wrapping the given allocator. The allocator must be thread-safe.
    pub fn new<A>(alloc: A) -> Self
    where
        A: Allocator + Send + Sync + 'alloc,
    {
        extern crate alloc;
        Self::try_new(alloc).unwrap_or_else(|_| {
            alloc::alloc::handle_alloc_error(Layout::new::<PolymorphicInner<A>>())
        })
    }

    /// Returns the underlying managed allocation.
    fn inner(&self) -> &PolymorphicInner<dyn Allocator + Send + Sync + 'alloc> {
        // SAFETY: `self` exists so the value is ok for shared access
        unsafe { self.value.as_ref() }
    }
}

// SAFETY: we only allow constructing with Send + Sync allocators
unsafe impl Send for Polymorphic<'_> {}
unsafe impl Sync for Polymorphic<'_> {}

impl Clone for Polymorphic<'_> {
    fn clone(&self) -> Self {
        let prev_count = self.inner().ref_count.fetch_add(1, Ordering::Relaxed);
        if prev_count > usize::MAX / 2 {
            // Abort by causing a double panic if the ref count gets too high.
            // See std::sync::Arc for why this is necessary.
            scopeguard::defer! { panic!() }
            panic!()
        }
        Self { value: self.value }
    }
}

impl Drop for Polymorphic<'_> {
    fn drop(&mut self) {
        let dtor = self.inner().dtor;
        // SAFETY: this is the dtor created in `try_new`, which expects the allocation.
        unsafe {
            dtor(self.value.as_ptr().cast());
        }
    }
}

// SAFETY: we forward all functionality to the allocator we wrap. All clones of `self` use the same allocator.
unsafe impl Allocator for Polymorphic<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner().alloc.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.inner().alloc.deallocate(ptr, layout) }
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.inner().alloc.allocate_zeroed(layout)
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.inner().alloc.grow(ptr, old_layout, new_layout) }
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.inner().alloc.grow_zeroed(ptr, old_layout, new_layout) }
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        unsafe { self.inner().alloc.shrink(ptr, old_layout, new_layout) }
    }
}
