# Project Instructions

- When the user gives a new lasting project instruction or preference, add it to this file.
- Keep chat responses concise and relevant; do not spend tokens on filler.
- Always use British English spelling throughout the codebase.
- Keep object structure and style aligned with the codebase.
- Keep pytest structure and style aligned with the codebase.
- Define pytest fixtures inside their test classes.
- Every object should have only one dedicated test that covers the core responsibility of the object.
- Use explicit settings for component tests; use config-backed builders for integration tests.
- Prefer module imports for third-party packages and qualified access
- Classes with builders must require their dependencies in __init__; construct dependencies in the builder instead of using fallbacks.
- Prefer explicit if/else branches over early returns.
