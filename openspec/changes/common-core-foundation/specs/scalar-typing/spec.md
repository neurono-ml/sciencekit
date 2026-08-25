## Purpose

Defines the library's numeric contract: a single sealed trait gathering the guarantees required from floating-point types used across all generic APIs, allowing generic dtypes as trait parameters without opening the surface to arbitrary external implementations.

## ADDED Requirements

### Requirement: Single sealed numeric trait
The library SHALL expose exactly one sealed floating-point trait aggregating the bounds required for numeric computation (arithmetic, copy, thread transfer and static dispatch), and every generic API accepting continuous numeric data SHALL use this trait as its bound — never loose bounds scattered around.

#### Scenario: Native float types satisfy the contract
- **WHEN** a generic algorithm is instantiated with the standard floating-point types supported by the library
- **THEN** instantiation compiles without additional user implementations

#### Scenario: External implementation is prevented
- **WHEN** an external crate tries to implement the numeric trait for its own type
- **THEN** compilation fails because the trait is sealed

### Requirement: Integers do not satisfy continuous contracts
Integer types SHALL NOT satisfy the numeric bound of continuous APIs; integer data enters the library through the dedicated integral/target representations defined at the data boundary.

#### Scenario: Integer where continuous is expected fails compilation
- **WHEN** code tries to instantiate a continuous estimator with an integer type as dtype
- **THEN** compilation fails with a bound violation
