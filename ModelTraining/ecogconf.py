#! /usr/bin/env python
# encoding: utf-8
"""
ecogconf
~~~~~~~~~~~~~~~~~~~~~~~~
Configuration for data directories and other nonsence

:copyright: 2014 by Griffin Milsap see AUTHORS for more details
:license: BSD, see LICENSE for more details
"""
import os
import json
import scipy.io as sio

################################### SAVE SELF ###################################
import shutil
def save_script_backup():
    
    """
    Automatically saving this entire script immediately when it's called.
    """
    
    # Creating the original and target file directories.
    original = r'/home/dan/Projects/PseudoOnlineTests_for_RTCoG/Scripts/Recent/ecogconf.py'
    target   = r'/mnt/shared/danprocessing/BACKUP/Projects/PseudoOnlineTests_for_RTCoG/Scripts/ecogconf.py'

    # Saving.
    shutil.copyfile(original, target)
    
# Immediately saving script.   
save_script_backup()

#################################################################################


class ECoGConfig( object ):

    local_config = None

    def __init__( self ):
        self.local_config = get_local_config()
        #print('local_config: ',self.local_config)
        #print('get_ecogdir: ',self.get_ecogdir())
        
        if not os.path.exists( self.get_ecogdir() ):
            print( "WARNING: ECOGDIR (%s) not found." % self.get_ecogdir() )
            print( "Set ECOGDIR as an environment variable, or edit the local config." )

    def get_ecogdir( self ):
        #print('local_config.ECOGDIR',self.local_config.ECOGDIR)
        return os.path.abspath( os.path.expanduser( self.local_config.ECOGDIR ) )
        

    def get_subjects( self ):
        try: return next( os.walk( self.get_ecogdir() ) )[1]
        except StopIteration: return []

    def get_datafiles( self, subject ):
        """ Returns a list of data files for the requested subject """
        hdf5_files = []
        subject_dir = self.get_subject_dir( subject )
        if subject_dir is not None:
            listing = os.listdir( subject_dir )
            
            for f in listing:
                fname = os.path.basename( f )
                if os.path.splitext( fname )[1] == '.hdf5':
                    hdf5_files.append( fname )

        hdf5_files = [ os.path.join( subject_dir, f ) for f in hdf5_files ]
        return hdf5_files

    def get_subject_dir( self, subject, create = False ):
        """
        If create == False, returns None if that subject doesn't exist
        If create == True, creates the subject dir if necessary.
        """
        subject_dir = os.path.join( self.get_ecogdir(), subject )

        if not os.path.exists( subject_dir ):
            if create: os.makedirs( subject_dir )
            else: return None
            
        return subject_dir

    def get_subject_info( self, subject ):
        """
        Returns a dictionary containing
            aux_channels: list of channels containing non-brain data
            bad_channels: list of bad channels
            grids: list of grid objects for plotting
        """
        meta_dir = self.get_meta_dir( subject )
        if meta_dir is None:
            return None

        subject_info = {}

        # channel_info
        channel_info_path = os.path.join( meta_dir, 'channel_info.json' )
        if os.path.exists( channel_info_path ):
            with open( channel_info_path, 'r' ) as channel_info_file:
                try:
                    channel_info = json.load( channel_info_file )
                    subject_info.update( channel_info )
                    subject_info['aux_channels'] = channel_info.get( 'auxChannels', None )
                    subject_info['bad_channels'] = channel_info.get( 'badChannels', None )
                except:
                    print( 'Error opening channel info' )
                    subject_info['aux_channels'] = None
                    subject_info['bad_channels'] = None
        else:
            print( 'Could not find channel info' )
            subject_info['aux_channels'] = None
            subject_info['bad_channels'] = None

        # grids
        grids_path = os.path.join( meta_dir, 'grids.json' )
        if os.path.exists( grids_path ):
            with open( grids_path, 'r' ) as grids_file:
                try:
                    subject_info['grids'] = json.load( grids_file )
                except:
                    subject_info['grids'] = None
        else:
            subject_info['grids'] = None

        return subject_info


    def get_analysis_dir( self, subject, create = False ):
        """
        If create == False, returns None if that subject doesn't exist
        If create == True, creates the subject dir if necessary.
        Returns the analysis subfolder within the patient's directory and
        creates it if it doesn't exist.
        """
        subject_dir = self.get_subject_dir( subject, create = create )
        if subject_dir is None: return None

        analysis_dir = os.path.join( subject_dir, 'analysis' )
        if not os.path.exists( analysis_dir ): os.makedirs( analysis_dir )
        return analysis_dir

    def get_reconstruction_dir( self, subject, create = False ):
        """
        If create == False, returns None if that subject doesn't exist
        If create == True, creates the subject dir if necessary.
        Returns the reconstruction subfolder within the patient's directory and
        creates it if it doesn't exist.
        """
        subject_dir = self.get_subject_dir( subject, create = create )
        if subject_dir is None: return None

        recon_dir = os.path.join( subject_dir, 'reconstruction' )
        if not os.path.exists( recon_dir ): os.makedirs( recon_dir )
        return recon_dir

    def get_meta_dir( self, subject, create = False ):
        """
        If create == False, returns None if that subject doesn't exist
        If create == True, creates the subject dir if necessary.
        Returns the meta subfolder within the patient's directory and
        creates it if it doesn't exist.
        """
        subject_dir = self.get_subject_dir( subject, create = create )
        if subject_dir is None: return None

        meta_dir = os.path.join( subject_dir, 'meta' )
        if not os.path.exists( meta_dir ): os.makedirs( meta_dir )
        return meta_dir

    def get_brainplotter( self, subject, brain = 'montage_low.png', locs = 'montage.csv' ):
        """
        Get a BrainPlotter object for the specified subject, readied with 
            brain and location data.

        NOTE: brain and locs are filenames relative to the subject's
            reconstruction directory.
        """
        from brainplotter import BrainPlotter
        recon_dir = self.get_reconstruction_dir( subject, create  = False )
        if recon_dir is None: return None

        brain_file = os.path.join( recon_dir, brain )
        locs_file = os.path.join( recon_dir, locs )
        return BrainPlotter( brain = brain_file, locs = locs_file )

    def analysis_filename( self, subject, filename, default_ext = '' ):
        """
        Given a subject, returns a full filename for the file filename within
        the subject's analysis directory.  Feel free to append subdirectories!
        Creates any necessary subdirectories when this is called, so be careful.
        """
        # Get the analysis output directory, creating it if necessary
        analysis_dir = self.get_analysis_dir( subject, create = True )
        result_dir = os.path.dirname( filename )
        result_dir = os.path.join( analysis_dir, result_dir )
        if not os.path.exists( result_dir ): os.makedirs( result_dir )
        
        # Ensure the result name is a properly formatted filename
        filename = os.path.basename( filename )
        name, ext = os.path.splitext( filename )
        if ext == '': ext = default_ext
        result_name = name + ext

        return os.path.join( result_dir, result_name )

    def query_datafiles( self, subject, query ):
        """
        Queries the list of data files for one containing the query string
        Returns a list of files
        """
        matching_files = []
        for f in self.get_datafiles( subject ):
            if query in os.path.basename( f ):
                matching_files.append( f )
        return matching_files

    def open_datafile( self, subject, query ):
        """
        Calls query_datafiles and opens the first result as an H5EEGFile
        This method is only for the extremely lazy
        """
        from h5eeg import H5EEGFile
        return H5EEGFile( self.query_datafiles( subject, query )[0], 'r' )

    def save_analysis( self, subject, filename, data_dict ):
        """
        Save analysis to a result file.
        Default behavior is to automatically overwrite if necessary.
        Choose a file type by specifying the file extension in the filename.

        data_dict should be formatted: { 'variable_name': variable, ... }

        Supported filetypes:
        * .mat (Matlab version 7 -- does not support more than 2 GB of data.
            This is a wrapper for scipy.io.savemat. May mangle lists of strings.
        * TODO: .npz (Uncompressed numpy array).
        """
        fname = self.analysis_filename( subject, filename, default_ext = '.mat' )

        # Save the file
        if os.path.splitext( fname )[1] == '.mat':
            sio.savemat( fname, data_dict )
        elif os.path.splitext( fname )[1] == '.npz':
            pass
        else: raise Exception( "Unknown data file format." )

    def load_analysis( self, subject, filename ):
        """
        Load a previously saved analysis result dictionary from a 
        file in the subject's analysis folder.

        Protip: Pass this into locals().update( ... ) for great justice.
        """

        fname = self.analysis_filename( subject, filename, default_ext = '.mat' )
        if not os.path.exists( fname ): raise FileNotFoundError()

        # Load the file
        data_dict = {}
        if os.path.splitext( fname )[1] == '.mat':
            data_dict = sio.loadmat( fname )
        elif os.path.splitext( fname )[1] == '.npz':
            pass
        else: raise Exception( "Unknown data file format." )

        return data_dict

CONFIG_MOD = 'ecogpy_config'
ENV_ECOGDIR = 'ECOGDIR'

def get_local_config():

    config_fname = CONFIG_MOD + '.py'
    
    #print('Config Mod: ',CONFIG_MOD)

    if not os.path.isfile( config_fname ):
        f = open( config_fname, 'w' )
        f.write( "#! /usr/bin/env/python\n" )
        f.write( "# You will need to restart your kernel after editing this file\n" )
        f.write( "# Edit the following variables to point to local paths\n" )
        ecogdir = os.getenv( ENV_ECOGDIR, '~/ecog' )
        f.write( "ECOGDIR = '%s'\n" % ecogdir )
        f.close()
        print( "Just created a config file for this analysis." )
        print( "You may want to edit %s and restart the kernel." % config_fname )

    return __import__( CONFIG_MOD )

