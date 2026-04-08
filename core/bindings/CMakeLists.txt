pybind11_add_module(_core py_aerial_manipulator.cpp)
target_link_libraries(_core PRIVATE aerial_manipulator_core)

install(TARGETS _core DESTINATION aerial_manipulator/models)
