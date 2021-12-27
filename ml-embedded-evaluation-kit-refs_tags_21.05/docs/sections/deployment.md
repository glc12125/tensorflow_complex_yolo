# Deployment

- [Deployment](#deployment)
  - [Fixed Virtual Platform](#fixed-virtual-platform)
    - [Setting up the MPS3 Arm Corstone-300 FVP](#setting-up-the-mps3-arm-corstone-300-fvp)
    - [Deploying on an FVP emulating MPS3](#deploying-on-an-fvp-emulating-mps3)
  - [MPS3 board](#mps3-board)
    - [MPS3 board top-view](#mps3-board-top-view)
    - [Deployment on MPS3 board](#deployment-on-mps3-board)

The sample application for Arm® *Ethos™-U55* can be deployed on two target platforms:

- A physical Arm MPS3 FPGA prototyping board

- An MPS3 FVP

Both implement the Arm® *Corstone™-300* design. For further information, please refer to:
[Arm Corstone-300](https://www.arm.com/products/iot/soc/corstone-300)

## Fixed Virtual Platform

The FVP is available publicly from the following page:
[Arm Ecosystem FVP downloads](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps).

Please ensure that you download the correct archive from the list under `Arm Corstone-300`. You need the one which:

- Emulates MPS3 board and *not* for MPS2 FPGA board,
- Contains support for Arm® *Ethos™-U55*.

### Setting up the MPS3 Arm Corstone-300 FVP

For the *Ethos-U55* sample application, please download the MPS3 version of the Arm® *Corstone™-300* model that contains
both the *Ethos-U55* and *Arm® Cortex®-M55*. The model is currently only supported on Linux-based machines.

To install the FVP:

- Unpack the archive.

- Run the install script in the extracted package:

    `./FVP_Corstone_SSE-300_Ethos-U55.sh`

- Follow the instructions to install the FVP to your required location.

### Deploying on an FVP emulating MPS3

This section assumes that the FVP has been installed (see
[Setting up the MPS3 Arm Corstone-300 FVP](#setting-up-the-mps3-arm-corstone-300-fvp))
to the home directory of the user: `~/FVP_Corstone_SSE-300_Ethos-U55`.

The installation, typically, has the executable under `~/FVP_Corstone_SSE-300_Ethos-U55/model/<OS>_<compiler-version>/`
directory. For the example below, we assume it is: `~/FVP_Corstone_SSE-300_Ethos-U55/models/Linux64_GCC-6.4`.

To run a use-case on the FVP, from the [Build directory](../sections/building.md#create-a-build-directory):

```commandline
~/FVP_Corstone_SSE-300_Ethos-U55/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55 -a ./bin/ethos-u-<use_case>.axf
telnetterminal0: Listening for serial connection on port 5000
telnetterminal1: Listening for serial connection on port 5001
telnetterminal2: Listening for serial connection on port 5002
telnetterminal5: Listening for serial connection on port 5003

    Ethos-U rev 0 --- Oct 13 2020 11:27:45
    (C) COPYRIGHT 2019-2020 Arm Limited
    ALL RIGHTS RESERVED
```

This also launches a telnet window with the standard output from the sample application. And also error log entries
containing information about the pre-built application version, TensorFlow Lite Micro library version used, and data
type. It also includes the input and output tensor sizes of the model that are compiled into the executable binary.

> **Note:** For details on the specific use-case, follow the instructions in the corresponding documentation.

After starting, the application outputs a menu and waits for the user-input from the telnet terminal.

For example, the image classification use-case can be started by using:

```commandline
~/FVP_Corstone_SSE-300_Ethos-U55/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55 -a ./bin/ethos-u-img_class.axf
```

![FVP](../media/fvp.png)

![FVP Terminal](../media/fvpterminal.png)

The FVP supports many command-line parameters, such as:

- Those passed by using `-C <param>=<value>`. The most important ones are:
  - `ethosu.num_macs`: Sets the *Ethos-U55* configuration for the model. Valid parameters are `32`, `64`, `256`, and the
    default one `128`. The number signifies the 8x8 MACs that are performed per cycle-count and that are available on
    the hardware.
  - `cpu0.CFGITCMSZ`: The ITCM size for the *Cortex-M* CPU. The size of ITCM is *pow(2, CFGITCMSZ - 1)* KB
  - `cpu0.CFGDTCMSZ`: The DTCM size for the *Cortex-M* CPU. The size of DTCM is *pow(2, CFGDTCMSZ - 1)* KB
  - `mps3_board.telnetterminal0.start_telnet`: Starts the telnet session if nothing connected.
  - `mps3_board.uart0.out_file`: Sets the output file to hold the data written by the UART. Use `'-'` to send all output
    to `stdout` and is empty by default).
  - `mps3_board.uart0.shutdown_on_eot`: Shut down the simulation when an `EOT (ASCII 4)` char is transmitted.
  - `mps3_board.visualisation.disable-visualisation`: Enables, or disables, visualization and is disabled by default.

  To start the model in `128` mode for *Ethos-U55*:

    ```commandline
    ~/FVP_Corstone_SSE-300_Ethos-U55/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55 -a ./bin/ethos-u-img_class.axf -C ethosu.num_macs=128
    ```

- `-l`: shows the full list of supported parameters

    ```commandline
    ~/FVP_Corstone_SSE-300_Ethos-U55/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55 -l
    ```

- `--stat`: prints some run statistics on simulation exit

    ```commandline
    ~/FVP_Corstone_SSE-300_Ethos-U55/models/Linux64_GCC-6.4/FVP_Corstone_SSE-300_Ethos-U55 --stat
    ```

- `--timelimit`: sets the number of wall clock seconds for the simulator to run, excluding startup and shutdown.

## MPS3 board

> **Note:**  Before proceeding, make sure that you have the MPS3 board powered on, and a USB A to B cable connected
> between your machine and the MPS3. The connector on the MPS3 is marked as "Debug USB".

![MPS3](../media/mps3.png)

### MPS3 board top-view

Once the board has booted, the micro SD card is enumerated as a mass storage device. On most systems, this is
automatically mounted. However, manual mounting is sometimes required.

Also, check for four serial-over-USB ports that are available for use through this connection. On Linux-based machines,
these would typically be */dev/ttyUSB\<n\>* to */dev/ttyUSB\<n+3\>*.

The default configuration for all of them is `115200`, `8/N/1`. So, 15200 Baud, 8 bits, no parity, and one stop bit,
with no flow control.

> **Note:** For Windows machines, extra FTDI drivers may be required for these serial ports to be available.

For more information on getting started with an MPS3 board, please refer to:
[MPS3 Getting Started](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/MPS3GettingStarted.pdf).

### Deployment on MPS3 board

> **Note:**: These instructions are valid only if the evaluation is being done using the MPS3 FPGA platform using
> `SSE-300`.

To run the application on MPS3 platform, you must first ensure that the platform has been set up using the correct
configuration.

For details on platform set-up, please see the relevant documentation. For the Arm `Corstone-300`, the PDF is available
here: [Arm Developer](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/DAI0547B_SSE300_PLUS_U55_FPGA_for_mps3.pdf?revision=d088d931-03c7-40e4-9045-31ed8c54a26f&la=en&hash=F0C7837C8ACEBC3A0CF02D871B3A6FF93E09C6B8).

For the MPS3 board, instead of loading the `axf` file directly, copy the executable blobs generated under the
`sectors/<use_case>` subdirectory to the micro SD card located on the board. Also, the `sectors/images.txt` file is used
by the MPS3 to understand which memory regions the blobs must be loaded into.

Once the USB A to USB B cable between the MPS3 and the development machine is connected, and the MPS3 board powered on,
the board enumerates as a mass storage device over this USB connection.

Depending on the version of the board you are using, there might be two devices listed. The device named `V2M-MPS3`, or
`V2MMPS3`, which is the `SD card`.

If the `axf` or `elf` file is within the ITCM load size limit, it can be copied into the FPGA memory directly without
having to break it down into separate load region-specific blobs. However, if the neural network models exceed this
size, you must use the following approach:

1. For example, the image classification use-case produces:

    ```tree
    ./bin/sectors/
        └── img_class
            ├── ddr.bin
            └── itcm.bin
    ```

    If the micro SD card is mounted at `/media/user/V2M-MPS3/`, then use:

    ```commandline
    cp -av ./bin/sectors/img_class/* /media/user/V2M-MPS3/SOFTWARE/
    ```

2. The `./bin/sectors/images.txt` file must be copied over to the MPS3. The exact location for the destination depends
   on the version of the MPS3 board and the application note for the bit file in use.

   For example, the revision C of the MPS3 board hardware uses an application note directory named `ETHOSU`, to replace the
   `images.txt` file, like so:

    ```commandline
    cp ./bin/sectors/images.txt /media/user/V2M-MPS3/MB/HBI0309C/ETHOSU/images.txt
    ```

3. Open the first serial port available from MPS3. For example, `/dev/ttyUSB0`. This can be typically done using
   minicom, screen, or Putty application. Make sure the flow control setting is switched off:

    ```commandline
    minicom --D /dev/ttyUSB0
    ```

    ```log
    Welcome to minicom 2.7.1
    OPTIONS: I18n
    Compiled on Aug 13 2017, 15:25:34.
    Port /dev/ttyUSB0, 16:05:34
    Press CTRL-A Z for help on special keys
    Cmd>
    ```

4. In another terminal, open the second serial port. For example: `/dev/ttyUSB1`:

    ```commandline
    minicom --D /dev/ttyUSB1
    ```

5. On the first serial port, issue a "reboot" command and then press the return key:

    ```commandline
    $ Cmd> reboot
    ```

    ```log
    Rebooting...Disabling debug USB..Board rebooting...

    ARM V2M-MPS3 Firmware v1.3.2
    Build Date: Apr 20 2018

    Powering up system...
    Switching on main power...
    Configuring motherboard (rev C, var A)...
    ```

    This goes on to reboot the board and prime the application to run by flashing the binaries into their respective
    FPGA memory locations. For example:

    ```log
    Reading images file \MB\HBI0309C\ETHOSU\images.txt
    Writing File \SOFTWARE\itcm.bin to Address 0x00000000

    ............

    File \SOFTWARE\itcm.bin written to memory address 0x00000000
    Image loaded from \SOFTWARE\itcm.bin
    Writing File \SOFTWARE\ddr.bin to Address 0x08000000

    ..........................................................................


    File \SOFTWARE\ddr.bin written to memory address 0x08000000
    Image loaded from \SOFTWARE\ddr.bin
    ```

6. When the reboot from previous step is completed, issue a reset command on the command prompt:

    ``` commandline
    $ Cmd> reset
    ```

    This triggers the application to start, and the output becomes visible on the second serial connection.

7. On the second serial port, the output is similar to that in section 2.2, is visible, like so:

    ```log
    INFO - Setting up system tick IRQ (for NPU)
    INFO - V2M-MPS3 revision C
    INFO - Application Note AN540, Revision B
    INFO - FPGA build 1
    INFO - Core clock has been set to: 32000000 Hz
    INFO - CPU ID: 0x410fd220
    INFO - CPU: Cortex-M55 r0p0
    ...
    ```

The next section of the documentation details: [Implementing custom ML application](customizing.md).
