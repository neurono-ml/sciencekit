-- Lua 5.4 smoke suite for the sciencekit native module (spec `lua-bindings`).
--
-- Run it against a built module without installing anything:
--
--   cargo build -p sciencekit_lua --release
--   SCIENCEKIT_LUA_LIB=target/release/libsciencekit_lua.so lua5.4 tests/run.lua
--
-- Or let LuaRocks install the module and call require("sciencekit") without
-- the environment variable:
--
--   luarocks make sciencekit-0.5.0-1.rockspec
--   lua5.4 tests/run.lua
--
-- Every estimator family the C ABI carries gets a fit/predict fixture:
-- linear regression and the standard scaler.

local function load_sciencekit()
  local path = os.getenv("SCIENCEKIT_LUA_LIB")
  if path == nil or path == "" then
    for _, candidate in ipairs({
      "target/release/libsciencekit_lua.so",
      "target/debug/libsciencekit_lua.so",
    }) do
      local file = io.open(candidate, "rb")
      if file ~= nil then
        file:close()
        path = candidate
        break
      end
    end
  end
  if path == nil or path == "" then
    return require("sciencekit")
  end
  local loader, message = package.loadlib(path, "luaopen_sciencekit")
  if loader == nil then
    error("cannot load the native module at " .. path .. ": " .. tostring(message))
  end
  return loader()
end

local sciencekit = load_sciencekit()

local function assert_near(actual, expected, label, tolerance)
  tolerance = tolerance or 1e-8
  if math.abs(actual - expected) > tolerance then
    error(string.format("%s: expected %.12g, got %.12g", label, expected, actual))
  end
end

local function assert_error_contains(callback, expected, label)
  local ok, message = pcall(callback)
  if ok then
    error(label .. ": expected an error")
  end
  message = tostring(message)
  if not message:find(expected, 1, true) then
    error(string.format("%s: expected %q in %q", label, expected, message))
  end
end

local tests = {}

function tests.module_metadata()
  assert(type(sciencekit.version) == "string", "version is a string")
  assert(sciencekit.version == "0.5.0", "version matches the crate")
  local abi = sciencekit.abi_version()
  assert(abi.major == 1 and abi.minor == 0, "ABI version is reported")
  assert(type(sciencekit.features()) == "number", "feature bitmask is reported")
end

function tests.linear_regression_family()
  local features = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 2.0, 3.0 } }
  local targets = { 1.0, 3.0, 4.0, 14.0 }
  local builder = sciencekit.linear_regression()
  assert(builder == builder:include_intercept(true), "setters chain")
  assert(builder == builder:backend("faer"), "backend chains")
  assert(builder.family == "linear-regression", "family is reported")
  local model = builder:fit(features, targets)
  assert(model.family == "linear-regression", "model family is reported")
  local predictions = model:predict(features)
  assert(#predictions == #targets, "one row per sample")
  assert(#predictions[1] == 1, "one column per prediction")
  for index = 1, #targets do
    assert_near(predictions[index][1], targets[index], "prediction " .. index)
  end
  assert_near(model:score(features, targets), 1.0, "R squared")
end

function tests.linear_regression_persistence()
  local features = { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { 2.0, 3.0 } }
  local targets = { 1.0, 3.0, 4.0, 14.0 }
  local path = os.tmpname()
  local model = sciencekit.linear_regression():fit(features, targets)
  model:save(path)
  local loaded = sciencekit.load(path)
  os.remove(path)
  assert(loaded.family == "linear-regression", "loaded family")
  assert_near(loaded:predict({ { 2.0, 3.0 } })[1][1], 14.0, "loaded prediction")
end

function tests.standard_scaler_family()
  local features = { { 0.0, 0.0 }, { 2.0, 2.0 }, { 4.0, 4.0 }, { 6.0, 6.0 } }
  local builder = sciencekit.standard_scaler()
  assert(builder == builder:with_mean(true), "with_mean chains")
  assert(builder == builder:with_std(true), "with_std chains")
  assert(builder.family == "standard-scaler", "family is reported")
  local model = builder:fit(features)
  local transformed = model:transform(features)
  assert(#transformed == 4, "one row per sample")
  assert(#transformed[1] == 2, "one column per feature")
  assert_near(transformed[1][1] + transformed[4][1], 0.0, "column is centered")
  assert_near(transformed[1][1], transformed[1][2], "columns share a scale")

  local centered = sciencekit.standard_scaler():with_mean(true):with_std(false)
  local offset = centered:fit(features):transform(features)
  assert_near(offset[1][1], -3.0, "with_std(false) only centers")
end

function tests.errors_are_structured()
  assert_error_contains(function()
    sciencekit.linear_regression():fit({ { 1.0, 2.0 }, { 3.0 } }, { 1.0, 2.0 })
  end, "sciencekit shape (2)", "ragged table")
  assert_error_contains(function()
    sciencekit.standard_scaler():fit({ { 1.0, "two" } })
  end, "sciencekit invalid-argument (3)", "non-numeric cell")
  assert_error_contains(function()
    local model = sciencekit.standard_scaler():fit({ { 1.0 }, { 2.0 } })
    model:predict({ { 1.0 } })
  end, "sciencekit unsupported (4)", "wrong family")
end

function tests.lifecycle_is_single_free()
  collectgarbage("collect")
  collectgarbage("collect")
  local baseline = sciencekit.live_handles()
  local model = sciencekit.linear_regression():fit({ { 1.0 }, { 2.0 }, { 3.0 } }, { 2.0, 4.0, 6.0 })
  model:close()
  model:close()
  assert(model.closed, "close marks the model closed")
  assert_error_contains(function()
    model:predict({ { 1.0 } })
  end, "sciencekit invalid-handle (1)", "closed model")
  model = nil
  collectgarbage("collect")
  collectgarbage("collect")
  assert(sciencekit.live_handles() == baseline, "the collector frees every handle once")
end

local failures = 0
local order = {
  "module_metadata",
  "linear_regression_family",
  "linear_regression_persistence",
  "standard_scaler_family",
  "errors_are_structured",
  "lifecycle_is_single_free",
}
for _, name in ipairs(order) do
  local ok, message = pcall(tests[name])
  if ok then
    print("ok   " .. name)
  else
    failures = failures + 1
    print("FAIL " .. name .. ": " .. tostring(message))
  end
end

if failures > 0 then
  error(string.format("%d Lua test(s) failed", failures))
end
print("sciencekit Lua suite passed")
