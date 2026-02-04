import os
from gi.repository import Gimp



def load_file(pdb, input_file):
    lfp = pdb.lookup_procedure('gimp-file-load')
    cfg = lfp.create_config()

    cfg.set_property('run-mode', Gimp.RunMode.NONINTERACTIVE)
    cfg.set_property('file', input_file)

    return lfp.run(cfg)



def load_file_layer(pdb, image, input_file):
    llfp = pdb.lookup_procedure('gimp-file-load-layer')
    cfg = llfp.create_config()

    cfg.set_property('run-mode', Gimp.RunMode.NONINTERACTIVE)
    cfg.set_property('image', image)
    cfg.set_property('file', input_file)

    return llfp.run(cfg)



def save_file(pdb, image, output_file):
    spp = pdb.lookup_procedure('file-png-export')
    cfg = spp.create_config()

    cfg.set_property('run-mode', Gimp.RunMode.NONINTERACTIVE)
    cfg.set_property('image', image)
    cfg.set_property('file', output_file)

    return spp.run(cfg)



def run():

    in1_name = os.getenv("GIMP_ATS_TEMP_0")
    in2_name = os.getenv("GIMP_ATS_TEMP_1")
    out_name = os.getenv("GIMP_ATS_TEMP_2")


    # Create necessary files.
    in1_file = Gio.File.new_for_path(in1_name)
    in2_file = Gio.File.new_for_path(in2_name)
    out_file = Gio.File.new_for_path(out_name)


    # Get PBD.
    pdb = Gimp.get_pdb()


    result = load_file(pdb, in1_file)
    error_code = result.index(0)


    # ErrorCode CHECK
    if (error_code == Gimp.PDBStatusType.SUCCESS):
        Gimp.message("Successfuly loaded a file.")
    else:
        raise RuntimeError("ERROR: loading a file.")


    image = result.index(1)
    result = load_file_layer(pdb, image, in2_file)
    error_code = result.index(0)


    # ErrorCode CHECK
    if (error_code == Gimp.PDBStatusType.SUCCESS):
        Gimp.message("Successfuly loaded a file as layer.")
    else:
        raise RuntimeError("ERROR: Loading a file as layer.")


    layer = result.index(1)
    layer.set_mode(Gimp.LayerMode.DIFFERENCE)
    image.insert_layer (layer, None, 0)


    result = save_file(pdb, image, out_file)
    error_code = result.index(0)


    # ErrorCode CHECK
    if (error_code == Gimp.PDBStatusType.SUCCESS):
        Gimp.message("Successfuly saved a file.")
    else:
        raise RuntimeError("ERROR: Saving a file.")


    # Dispose the file.
    layer.delete()
    image.delete()

    return 0

run()
