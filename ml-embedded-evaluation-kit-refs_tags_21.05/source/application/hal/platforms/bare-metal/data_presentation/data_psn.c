/*
 * Copyright (c) 2021 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "data_psn.h"

#include "bsp.h"
#include "lcd_img.h"

#include <assert.h>
#include <string.h>

int data_psn_system_init(data_psn_module* module)
{
    assert(module);

    /* LCD output supported. */
    module->system_init = lcd_init;
    module->present_data_image = lcd_display_image;
    module->present_data_text = lcd_display_text;
    module->present_box = lcd_display_box;
    module->set_text_color = lcd_set_text_color;
    module->clear = lcd_clear;
    strncpy(module->system_name, "lcd", sizeof(module->system_name));
    module->inited =  !module->system_init();
    return !module->inited;
}

int data_psn_system_release(data_psn_module* module)
{
    assert(module);
    module->inited = 0;
    return 0;
}
