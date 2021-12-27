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
#ifndef DEVICE_MPS3_H
#define DEVICE_MPS3_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cmsis.h"      /* CMSIS device header. */
#include "smm_mps3.h"   /* Memory map for MPS3. */

#include <stdio.h>

typedef struct _CMSDK_UART_TypeDef_
{
    __IO uint32_t  DATA;        /* Offset: 0x000 (R/W) Data Register.    */
    __IO uint32_t  STATE;       /* Offset: 0x004 (R/W) Status Register.  */
    __IO uint32_t  CTRL;        /* Offset: 0x008 (R/W) Control Register. */

    union {
    __I  uint32_t  INTSTATUS;   /* Offset: 0x00C (R/ ) Interrupt Status Register. */
    __O  uint32_t  INTCLEAR;    /* Offset: 0x00C ( /W) Interrupt Clear Register. */
    };
    __IO uint32_t  BAUDDIV;     /* Offset: 0x010 (R/W) Baudrate Divider Register. */

} CMSDK_UART_TypeDef;

#define CMSDK_UART0             ((CMSDK_UART_TypeDef *)CMSDK_UART0_BASE)

/* CMSDK_UART DATA Register Definitions. */
#define CMSDK_UART_DATA_Pos               0                                             /* CMSDK_UART_DATA_Pos: DATA Position. */
#define CMSDK_UART_DATA_Msk              (0xFFul << CMSDK_UART_DATA_Pos)                /* CMSDK_UART DATA: DATA Mask. */

/* CMSDK_UART STATE Register Definitions. */
#define CMSDK_UART_STATE_RXOR_Pos         3                                             /* CMSDK_UART STATE: RXOR Position. */
#define CMSDK_UART_STATE_RXOR_Msk         (0x1ul << CMSDK_UART_STATE_RXOR_Pos)          /* CMSDK_UART STATE: RXOR Mask. */

#define CMSDK_UART_STATE_TXOR_Pos         2                                             /* CMSDK_UART STATE: TXOR Position. */
#define CMSDK_UART_STATE_TXOR_Msk         (0x1ul << CMSDK_UART_STATE_TXOR_Pos)          /* CMSDK_UART STATE: TXOR Mask. */

#define CMSDK_UART_STATE_RXBF_Pos         1                                             /* CMSDK_UART STATE: RXBF Position. */
#define CMSDK_UART_STATE_RXBF_Msk         (0x1ul << CMSDK_UART_STATE_RXBF_Pos)          /* CMSDK_UART STATE: RXBF Mask. */

#define CMSDK_UART_STATE_TXBF_Pos         0                                             /* CMSDK_UART STATE: TXBF Position. */
#define CMSDK_UART_STATE_TXBF_Msk         (0x1ul << CMSDK_UART_STATE_TXBF_Pos )         /* CMSDK_UART STATE: TXBF Mask. */

/* CMSDK_UART CTRL Register Definitions. */
#define CMSDK_UART_CTRL_HSTM_Pos          6                                             /* CMSDK_UART CTRL: HSTM Position. */
#define CMSDK_UART_CTRL_HSTM_Msk          (0x01ul << CMSDK_UART_CTRL_HSTM_Pos)          /* CMSDK_UART CTRL: HSTM Mask. */

#define CMSDK_UART_CTRL_RXORIRQEN_Pos     5                                             /* CMSDK_UART CTRL: RXORIRQEN Position. */
#define CMSDK_UART_CTRL_RXORIRQEN_Msk     (0x01ul << CMSDK_UART_CTRL_RXORIRQEN_Pos)     /* CMSDK_UART CTRL: RXORIRQEN Mask. */

#define CMSDK_UART_CTRL_TXORIRQEN_Pos     4                                             /* CMSDK_UART CTRL: TXORIRQEN Position. */
#define CMSDK_UART_CTRL_TXORIRQEN_Msk     (0x01ul << CMSDK_UART_CTRL_TXORIRQEN_Pos)     /* CMSDK_UART CTRL: TXORIRQEN Mask. */

#define CMSDK_UART_CTRL_RXIRQEN_Pos       3                                             /* CMSDK_UART CTRL: RXIRQEN Position. */
#define CMSDK_UART_CTRL_RXIRQEN_Msk       (0x01ul << CMSDK_UART_CTRL_RXIRQEN_Pos)       /* CMSDK_UART CTRL: RXIRQEN Mask. */

#define CMSDK_UART_CTRL_TXIRQEN_Pos       2                                             /* CMSDK_UART CTRL: TXIRQEN Position. */
#define CMSDK_UART_CTRL_TXIRQEN_Msk       (0x01ul << CMSDK_UART_CTRL_TXIRQEN_Pos)       /* CMSDK_UART CTRL: TXIRQEN Mask. */

#define CMSDK_UART_CTRL_RXEN_Pos          1                                             /* CMSDK_UART CTRL: RXEN Position. */
#define CMSDK_UART_CTRL_RXEN_Msk          (0x01ul << CMSDK_UART_CTRL_RXEN_Pos)          /* CMSDK_UART CTRL: RXEN Mask. */

#define CMSDK_UART_CTRL_TXEN_Pos          0                                             /* CMSDK_UART CTRL: TXEN Position. */
#define CMSDK_UART_CTRL_TXEN_Msk          (0x01ul << CMSDK_UART_CTRL_TXEN_Pos)          /* CMSDK_UART CTRL: TXEN Mask. */

/* CMSDK_UART INTSTATUS\INTCLEAR Register Definitions. */
#define CMSDK_UART_INT_RXORIRQ_Pos        3                                             /* CMSDK_UART INT: RXORIRQ Position. */
#define CMSDK_UART_INT_RXORIRQ_Msk        (0x01ul << CMSDK_UART_INT_RXORIRQ_Pos)        /* CMSDK_UART INT: RXORIRQ Mask. */

#define CMSDK_UART_INT_TXORIRQ_Pos        2                                             /* CMSDK_UART INT: TXORIRQ Position. */
#define CMSDK_UART_INT_TXORIRQ_Msk        (0x01ul << CMSDK_UART_INT_TXORIRQ_Pos)        /* CMSDK_UART INT: TXORIRQ Mask. */

#define CMSDK_UART_INT_RXIRQ_Pos          1                                             /* CMSDK_UART INT: RXIRQ Position. */
#define CMSDK_UART_INT_RXIRQ_Msk          (0x01ul << CMSDK_UART_INT_RXIRQ_Pos)          /* CMSDK_UART INT: RXIRQ Mask. */

#define CMSDK_UART_INT_TXIRQ_Pos          0                                             /* CMSDK_UART INT: TXIRQ Position. */
#define CMSDK_UART_INT_TXIRQ_Msk          (0x01ul << CMSDK_UART_INT_TXIRQ_Pos)          /* CMSDK_UART INT: TXIRQ Mask. */

/* CMSDK_UART BAUDDIV Register Definitions. */
#define CMSDK_UART_BAUDDIV_Pos            0                                             /* CMSDK_UART BAUDDIV: BAUDDIV Position. */
#define CMSDK_UART_BAUDDIV_Msk           (0xFFFFFul << CMSDK_UART_BAUDDIV_Pos)

/**
 * @brief   Gets the core clock set for MPS3.
 * @return  Clock value in Hz.
 **/
uint32_t GetMPS3CoreClock(void);

#ifdef __cplusplus
}
#endif

#endif /* DEVICE_MPS3_H */
