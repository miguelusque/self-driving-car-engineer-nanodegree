# How to decompress

The files have been splitted using the following commands:

- split -b50mb mac-sim.app.zip "mac-sim.app.zip."
- split -b50mb linux-sim.zip "linux-sim.zip."
- split -b50mb windows-sim.zip "windows-sim.zip." 

In order to join them, any of the following commands:

- cat mac-sim.app.zip.* > mac-sim.app.zip
- cat linux-sim.zip.* > linux-sim.zip
- cat windows-sim.zip.* > windows-sim.zip

Finally, unzip the simulator using this:

- unzip mac-sim.app.zip