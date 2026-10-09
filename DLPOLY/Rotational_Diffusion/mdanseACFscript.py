# -*- coding: utf-8 -*-
########################################################
# MDANSE Angular Correlation Function Batch Script (MTO)
# Compatible with Python 2.7
########################################################

from MDANSE import REGISTRY

################################################################
# User Parameters — Change these as needed
################################################################

trajectory_path = u'C:\\Users\\gd478\\Documents\\phd_yr4\\MD\\correct\\12percwateroneperc5fu\\200ps_rot\\FIELD.nc'
output_folder   = u'C:\\Users\\gd478\\Documents\\phd_yr4\\MD\\correct\\12percwateroneperc5fu\\200ps_rot\\rot\\'

axes            = ['1', '2', '3']   # axis definitions from MDANSE GUI
segment_length  = 100               # number of frames per segment
segment_step    = 100               # overlap step (start increment)
final_frame     = 2000              # last frame index
frame_stride    = 1                 # frame stride within a segment
running_mode    = ('monoprocessor',)
per_axis        = False

################################################################
# Automated Loop — Multiple Time Origins (MTO)
################################################################

segment_count = 0

for start in range(0, final_frame - segment_length + 1, segment_step):
    end = start + segment_length - 1

    for axis in axes:
        segment_count += 1
        output_file = output_folder + unicode(segment_count)  # Python 2: use unicode()

        parameters = {
            'axis_selection': axis,
            'frames': (start, end, frame_stride),
            'output_files': (output_file, (u'hdf',)),
            'per_axis': per_axis,
            'running_mode': running_mode,
            'trajectory': trajectory_path
        }

        print "Running ACF for axis %s, frames %d-%d, output %d.hdf" % (axis, start, end, segment_count)

        ac = REGISTRY['job']['ac']()
        ac.run(parameters, status=True)

print "\nCompleted %d ACF analyses using the MTO method.\n" % segment_count
