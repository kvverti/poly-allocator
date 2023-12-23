Defines a polymorphic allocator type. This crate provides the type `Polymorphic<'alloc>`, a uniform representation
for an allocator of any type, like `dyn Allocator + 'alloc`, but which can be freely cloned and stored. Under the hood,
the backing allocator is stored in a reference-counted allocation.