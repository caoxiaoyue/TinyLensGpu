# Legacy Paper Scripts

**Status**: Archived - No longer maintained

These scripts use the legacy ModelParser/Profile/Simulator system which has been replaced by the new Caskade-based implementation.

## Contents

- **slacs/**: Scripts for SLACS (Sloan Lens ACS Survey) lens analysis
- **mock_csst/**: Mock CSST (Chinese Space Station Telescope) data analysis
- **benchmark/**: Performance benchmark scripts

## Why Archived?

The codebase has migrated to a new Caskade-based architecture which provides:
- Better modularity and maintainability
- Automatic parameter management
- Improved batch processing
- Cleaner separation of concerns

These legacy scripts are preserved for reference only and are not guaranteed to work with the current codebase.

## If You Need to Run These Scripts

1. The legacy code (ModelParser, Profile, Simulator, RunModel) has been removed from the main codebase
2. To run these scripts, you would need to check out an earlier version of the repository before the Caskade migration
3. For new work, please use the Caskade-based system (see demos in `paper/demo/`)

## Migration Information

See the main project documentation for migration guides:
- [MIGRATION_GUIDE.md](../../MIGRATION_GUIDE.md) - How to migrate from legacy to Caskade
- [CASKADE_GUIDE.md](../../CASKADE_GUIDE.md) - Getting started with Caskade system
- [CASKADE_API.md](../../CASKADE_API.md) - API reference for Caskade models

---

**Last Updated**: 2025-12-17
**Legacy Code Removal Date**: 2025-12-17
