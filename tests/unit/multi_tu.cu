#include <opworks>

// Force header-defined non-template kernels into a second translation unit.
void second_translation_unit() {
    opworks::ops::softmax(nullptr, nullptr, 0, 3);
    opworks::ops::layer_norm(nullptr, nullptr, nullptr, nullptr, 0, 3);
}
