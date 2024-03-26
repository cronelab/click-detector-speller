#! /usr/bin/env python
# encoding: utf-8
"""
h5eeg
~~~~~~~~~~~~~~~~~~~~~~~~
Wraps an h5py.File that uses the h5eeg format.

Barebones File Format:
* *`h5eeg/`* -- *Group* containing the data for this file.
    * Attribute: `subject` -- Subject identifier (optional)
    * Attribute: `experiment` -- Experiment identifier (optional)
    * Attribute: `datetime` -- Human readable date/time of start of recording (optional)
    * Attribute: `timestamp` -- POSIX formatted timestamp of start of recording (optional)
    * **`eeg`** -- **Dataset** (Samples by Channels) containing electrophysiological data.
        * Attribute: `labels` -- One dimensional array of strings for channel labels
        * Attribute: `offsets` -- One dimensional array of offsets in digital A/D units
        * Attribute: `gains` -- One dimensional array of gains to convert channels to muV
        * Attribute: `rate` -- Sampling Rate
    * **`aux`** -- **Dataset** (Samples by Channels) containing non-electrophysiologial 
            data used to mark/give meaning to the **`eeg`** dataset.
        * Attribute: `labels` -- One dimensional array of strings for auxiliary channel labels.
        * Attribute: `rate` -- Sampling Rate
    * **`events`** -- **Dataset** (Compound Datatype) containing indices of events in 
            the **`eeg`** dataset.

Example:
    from h5eeg import open_h5eeg
    group = open_h5eeg( 'somedir/somefile.hdf5' )
    group.eeg()[:,'FTG22'] # Data access to named channel FTG22
    group.aux()[:,'AINP2'] # Data access to named auxiliary channel AINP2 (Mic?)
    group.events().query_events( 'AH_SYLLABLE' ) # Query stored events

    # Plot all data during events containing the word 'AH' in FTG22
    plot( group.eeg()[ group.events().query( 'AH' ), 'FTG22' ] )

Check out test() for a more detailed example on how to use h5eeg to create/use
electrophysiological data stored in hdf5 files.

More information can be found on the wiki: 
https://github.com/griffinmilsap/Analysis/wiki/H5EEG

Dependencies: h5py, numpy (pylab/matplotlib for tests)

This file is a self-sufficient interface to H5EEG files and can be safely copied
into other projects separate from the rest of the python code in this repository.
Please see COPYING for license details when including this source code in your
own project.

:copyright: 2014 by Griffin Milsap, Alcimar Soares see AUTHORS for more details
:license: BSD, see LICENSE for more details
"""
import os
import sys
import time
import argparse
import datetime
import h5py as h5
import numpy as np





################################### SAVE SELF ###################################
import shutil
def save_script_backup():
    
    """
    Automatically saving this entire script immediately when it's called.
    """
    
    # Creating the original and target file directories.
    original = r'/home/dan/Projects/PseudoOnlineTests_for_RTCoG/Scripts/Recent/h5eeg.py'
    target   = r'/mnt/shared/danprocessing/BACKUP/Projects/PseudoOnlineTests_for_RTCoG/Scripts/h5eeg.py'

    # Saving.
    shutil.copyfile(original, target)
    
# Immediately saving script.   
save_script_backup()

#################################################################################





class H5EEGFile( object ):
    
    DEFAULT_GROUP = u'h5eeg'

    h5file = None
    group = None
    
    def __init__( self, *args, **kwargs ):
        """
        Same arguments has h5py.File, but you can also specify a group argument.
        By default group = 'h5eeg', but you can change this to open different groups
        in the HDF5 file.  

        If the file or the requested group doesn't exist, this constructor will attempt
        to create it, and any necessary directories in the process, unless mode = 'r'
        in which case, an exception will be raised.

        Keyword Arguments:
        group -- The name of the H5EEG Group within the root of the file.
            (Default: 'h5eeg')
        subject -- If a new group is being created, specify the subject ID here.
            Ignored unless a new group is being created
            (Default: 'unknown')
        experiment -- If a new group is being created, specify the experiment here.
            Ignored unless a new group is being created
            (Default: 'unknown')
        timestamp -- If a new group is being created, specify the timestamp here.
            Ignored unless a new group is being created
            (Default: right now)
        """
        # Determine what group to load from the file
        group = self.DEFAULT_GROUP
        if 'group' in kwargs:
            group = kwargs[ 'group' ]
        h5filename = args[0]

        # Load the file and return the H5EEGGroup if we can
        if os.path.isfile( h5filename ):
            self.h5file = h5.File( *args, **kwargs )
            if group in self.h5file.keys():
                self.group = H5EEGGroup( self.h5file.require_group( group ) )
                return
            else: self.close() # We don't have the group, so we'll create it
        
        # If we got here, the file doesn't exist or the requested group
        # doesn't exist.  We need to open the file with write access.
        # If the user adamantly wanted to keep things read-only, we should 
        # throw an exception that we can't proceed without write access.
        if ( len( args ) >= 2 and args[1] == 'r' ) or kwargs.get( 'mode', 'a' ) == 'r':
            raise ValueError( "Can not proceed without write access" )

        createKwargs = { 'name': group }
        createKwargs[ 'subject' ] = None
        if 'subject' in kwargs: 
            createKwargs[ 'subject'] = kwargs[ 'subject' ]
        createKwargs[ 'timestamp' ] = None
        if 'timestamp' in kwargs:
            createKwargs[ 'timestamp' ] = kwargs[ 'timestamp' ]
        createKwargs[ 'experiment' ] = None
        if 'experiment' in kwargs:
            createKwargs[ 'experiment' ] = kwargs[ 'experiment' ]

        # Create the directory if necessary
        file_dir = os.path.dirname( os.path.abspath( h5filename ) )
        if not os.path.exists( file_dir ):
            os.mkdir( file_dir )

        # Regardless of what the user wanted, we now need write permissions
        kwargs.pop( 'mode', None ) # Default mode is 'a'
        self.h5file = h5.File( h5filename, **kwargs )
        self.group = H5EEGGroup.create( self, **createKwargs )

    def flush( self ):
        """ Flush the file and write changes to disk """
        if self.h5file is not None: self.h5file.flush()

    def close( self ):
        """ Close the file and void all references """
        if self.h5file is not None: self.h5file.close()
        self.group = None

class H5EEGGroup( object ):
    """
    H5EEGGroup is the master class for a data group within an h5eeg bundle.
    This class represents the highest level functional element within h5eeg.

    Although the default group for a valid h5eeg file is named 'h5eeg', other
    groups may be interpreted as an H5EEGGroup provided they follow the proscribed
    format.  This is done by initalizing H5EEGGroup with an h5py.Group object.
    Example: H5EEGGroup( file[ 'another_group' ] ).

    This class assumes that the group is of a valid format.  Static 'create' 
    methods exist for H5EEGDataset, H5EEGAuxDataset, and H5EEGEvents, which enable
    a user to create their own H5EEGFile in the proper format using built-in
    functionality.
    """

    # Identifiers for H5EEGFile Attributes
    SUBJECT = u'subject'
    EXPERIMENT = u'experiment'
    TIMESTAMP = u'timestamp'
    DATETIME = u'datetime'

    # Group object
    group = None

    def __init__( self, group ):
        """ Interpret the input group as an H5EEGGroup """
        self.group = group

    def get_subject( self ):
        """ Get the subject name. Returns None if it doesn't exist. """
        if self.SUBJECT in self.group.attrs.keys():
            return self.group.attrs[ self.SUBJECT ]
        return None

    def set_subject( self, subject ):
        """ Set the subject name."""
        self.group.attrs[ self.SUBJECT ] = subject

    def get_experiment( self ):
        """ Get the experiment name. Returns None if it doesn't exist. """
        if self.EXPERIMENT in self.group.attrs.keys():
            return self.group.attrs[ self.EXPERIMENT ]
        return None

    def set_experiment( self, experiment ):
        """ Set the experiment name """
        self.group.attrs[ self.EXPERIMENT ] = experiment

    def get_timestamp( self ):
        """
        Retrieve a POSIX timestamp for the start of the recording
        Returns None if it doesn't exist.
        """
        if self.TIMESTAMP in self.group.attrs.keys():
            return self.group.attrs[ self.TIMESTAMP ]
        return None

    def set_timestamp( self, timestamp ):
        """ Set a POSIX timestamp for the start of the recording. """
        self.group.attrs[ self.TIMESTAMP ] = timestamp

    def get_datetime( self ):
        """
        Retreive a human readable timestamp for the start of the recording.
        Returns None it it doesn't exist.
        """
        timestamp = self.get_timestamp();
        if timestamp is not None:
            time = datetime.datetime.fromtimestamp( int( timestamp ) )
            return time.strftime('%Y-%m-%dT%H:%M:%S')
        return None

    def update_datetime( self ):
        """ Create/Update the 'datetime' attribute from the 'timestamp' attribute. """
        datetime = self.get_datetime();
        if datetime is not None:
            self.group.attrs[ self.DATETIME ] = datetime;

    # Access to internal datasets
    def eeg( self ):
        """
        Retrieve an H5EEGDataset object for the eeg dataset
        Returns None if this object doesn't exist.
        """
        if H5EEGDataset.NAME in self.group.keys():
            return H5EEGDataset( self.group[ H5EEGDataset.NAME ] )
        return None

    def aux( self ):
        """
        Retrieve a H5EEGAuxDataset object for the auxiliary dataset
        Returns None if this object doesn't exist.
        """
        if H5EEGAuxDataset.NAME in self.group.keys():
            return H5EEGAuxDataset( self.group[ H5EEGAuxDataset.NAME ] )
        return None

    def events( self ):
        """
        Retrieve an H5EEGEvents object for the events in this group 
        Returns None if this object doesn't exist.
        """
        if H5EEGEvents.NAME in self.group.keys():
            return H5EEGEvents( self.group[ H5EEGEvents.NAME ] )
        return None

    # Easy access to group
    def __getitem__( self, val ):
        return self.group.__getitem__( val )

    def __setitem__( self, val, item ):
        self.group.__setitem__( val, item )

    @staticmethod
    def create( h5eegfile, subject = None, experiment = None, timestamp = None, 
        name = H5EEGFile.DEFAULT_GROUP ):
        """
        Given a writable H5EEGFile object, this method creates the 'h5eeg' group
        and creates an empty events structure within it.

        If 'h5eeg' already exists, this method returns a handle to that existing
        group, after making changes to the attributes as specified in the parameters.

        Keyword Arguments:
        h5file -- writable h5py.File object (or h5py.Group)
        subject -- name of the subject (Default: 'unknown')
        experiment -- name of the experiment (Default: 'unknown')
        timestamp -- POSIX timestamp for the start of the recording (Default: now)

        Returns H5EEGGroup
        """
        if subject is None: subject = u'unknown'
        if experiment is None: experiment = u'unknown'
        if timestamp is None: timestamp = time.time()

        group = H5EEGGroup( h5eegfile.h5file.require_group( name ) )
        H5EEGEvents.create( group )
        group.set_subject( subject )
        group.set_experiment( experiment )
        group.set_timestamp( timestamp )
        group.update_datetime()
        return group

class LabeledTimeseries( object ):
    """
    Base class for H5EEGDataset and H5EEGAuxDataset.  This class implements
    functionality for retreiving data from named channels

    Functionality Not Implemented Yet:
        Translating ranges specified in time units to sample indices.
    """

    LABELS = u'labels'
    RATE = u'rate'

    dataset = None
    label_dict = dict()

    def __init__( self, dataset ):
        """ Interpret the input dataset as a LabeledTimeseries """
        if isinstance( dataset, LabeledTimeseries ):
            dataset = dataset.dataset
        self.dataset = dataset
        labels = self.get_labels()
        self._populate_label_dict( labels )

    # Getters and Setters
    def get_labels( self ):
        """
        Return the labels in this dataset.
        Returns None if no labels have been set.
        """
        if self.LABELS in self.dataset.attrs.keys():
            return self.dataset.attrs[ self.LABELS ].astype( str )
        return None

    def set_labels( self, labels ):
        """ Set the labels in this dataset """
        self.dataset.attrs[ self.LABELS ] = [ l.encode( 'utf8' ) for l in labels ]
        self._populate_label_dict( labels )

    def get_rate( self ):
        """
        Get the sampling rate (samples per second)
        Returns None if this attribute has not been defined.
        """
        if self.RATE in self.dataset.attrs.keys():
            return self.dataset.attrs[ self.RATE ]
        return None

    def sec_to_samp( self, sec ): return int( sec * self.get_rate() )
    def samp_to_sec( self, samp ): return samp / float( self.get_rate() )

    def set_rate( self, rate ):
        """ Set the sampling rate (samples per second) """
        self.dataset.attrs[ self.RATE ] = rate

    def get_num_samples( self ):
        """ Get the number of samples in this dataset """
        return self.dataset.len()

    def ch_indices( self, channels = None, exclude = None ):
        """
        Create a list of channel indices based on inclusion or exclusion
        If channels = exclude = None, returns all channel indices
        If channels is specified, a subset of channels is returned
        If exclude is specified, those channels are *not* returned

        NOTE: If a channel is exactly specified channels and exclude_ch,
        the corresponding channel index will not be returned.

        Both channels and exclude can be:
            * string filter (e.g. 'PMIC1' includes PMIC1, PMIC10, PMIC11, ...)
            * iterable (list of exact channel matches as strings and indices as ints)
            * None (No effect)

        Return Value: a list of channel indices corresponding to the query.
        
        Keyword Arguments:
            channels -- A filter or list of channels to include.
            exclude -- A filter or list of channels to exclude.
        """
        # Return all channel indices by default
        ret = list( self.label_dict.values() )

        # Define a lambda to help us translate mixed lists of strings and indices.
        to_idx = lambda ch: self.label_dict[ ch ] if isinstance( ch, str ) else ch

        if channels is not None:
            if isinstance( channels, str ):
                ret = [ self.label_dict[ ch ] for ch in self.get_labels() if channels in ch ]
            else: ret = [ to_idx( ch ) for ch in channels ]

        if exclude is not None:
            rm_ch = []
            if isinstance( exclude, str ):
                rm_ch = [ self.label_dict[ ch ] for ch in self.get_labels() if exclude in ch ]
            else: rm_ch = [ to_idx( ch ) for ch in exclude ]
            for ch in rm_ch: 
                if ch in ret: 
                    ret.remove( ch )

        return sorted( ret )


    # Plot a segment of data.  If start or duration are specified
    def plot( self, start = 0, duration = 5.0, channels = None, title = None, exclude = None ): 
        """ 
        Plot a segment of data.  Start and duration can be specified in samples
        or in time-units.

        Keyword Arguments:
            start/duration -- Specify start and duration of time for plotting.
                Note: If this is specified as an int, it refers to samples.
                Note: If this is specified as a float, it refers to seconds.
            title -- Specifies a title for the plot.
            channels/exclude -- Specify channels to include and/or exclude.
                Note: See documentation for ch_indices
        """
        import pylab as pl
        from matplotlib.collections import LineCollection
        from matplotlib.colors import colorConverter

        # Translate the times to samples
        start = self.as_samples( start )
        duration = self.as_samples( duration )
        end = start + duration

        # Translate input 
        ch_idx = self.ch_indices( channels = channels, exclude = exclude )

        # Grab the data
        data = self[ start:end, ch_idx ]
        rate = float( self.get_rate() )
        t = np.linspace( start / rate, end / rate, data.shape[0] )

        # Plot the data
        fig = pl.figure( self.dataset.name, figsize = ( 8.0, 0.5 * len( ch_idx ) ) )
        ax = fig.gca()

        pl.xlim( min( t ), max( t ) )
        pl.xlabel( 'Time (s)' )

        dmin = data.min()
        dmax = data.max()
        drange = ( dmax - dmin ) * 0.7 # Crowding
        pl.ylim( dmin, ( len( ch_idx ) - 1 ) * drange + dmax )

        ticklocs = []
        segs = []
        for i in range( len( ch_idx ) ):
            segs.append( np.hstack( ( t[ :, np.newaxis ], data[ :, i, np.newaxis ] ) ) )
            ticklocs.append( i  * drange )

        offsets = np.zeros( ( len( ch_idx ), 2 ), dtype = float )
        offsets[ :, 1 ] = ticklocs

        colors = [ colorConverter.to_rgba(c) for c in ('k','k','k','k','r') ]
        lines = LineCollection( segs, offsets = offsets, transOffset = None )
        ax.add_collection( lines )
        lines.set_color( colors )

        ax.set_yticks( ticklocs )
        ax.set_yticklabels( [ self.get_labels()[ idx ] for idx in ch_idx ] )
        ax.invert_yaxis()
        ax.grid( True )

        if title is not None:
            pl.title( title )

    # Plot an event in the LTS
    def plot_event( self, event, duration = 5.0, channels = None, title = None, exclude = None ):
        """ 
        Plot an event.  See documentation for plot(...) for more info. 
        NOTE: If an event's duration is '0', the duration supplied by the
        duration keyword argument will be used instead.  Otherwise, the duration
        keyword argument is ignored.
        """
        plot_duration = event[ 'duration' ]
        if plot_duration == 0: plot_duration = duration
        if title is None: title = event[ 'name' ]
        self.plot( event[ 'start_idx' ], plot_duration, 
            channels = channels, title = title, exclude = exclude ) 

    # Translates a slice with channel labels to a slice that h5py.dataset can retreive.
    def _translate_slice( self, val ):
        """ Translate the input slice from __get/setitem__ to sample indices. """
        # Initialize return values (THIS ASSUMES CHANNEL LABELS APPLY TO DIM[1])
        ret_t = val[0];
        ret_c = val[1];

        # TODO: Devise a way of dealing with times
        
        # Translate event into time slice
        if type( ret_t ) == np.void and ret_t.dtype == H5EEGEvents.EVENT_DTYPE:
            start_idx = ret_t[ H5EEGEvents.START ]
            duration = ret_t[ H5EEGEvents.DURATION ]
            ret_t = slice( start_idx, start_idx + duration )

        # If we have information about the channel labels, perform conversions.
        if ret_c is not None and len( self.label_dict ) != 0:

            # Convert a singular channel name into a channel index
            if isinstance( ret_c, str ):
                ret_c = self.label_dict[ ret_c ]
            elif isinstance( ret_c, ( list, tuple, np.ndarray ) ):
                ret_c = self.ch_indices( channels = ret_c )

        return ( ret_t, ret_c )

    def _populate_label_dict( self, labels = None ):
        """ Populate a dictionary to translate channel names to indices. """
        self.label_dict = dict()
        if labels is None: labels = self.get_labels()
        if labels is not None:
            for idx, label in enumerate( labels ):
                self.label_dict[ label ] = idx

    # Easy access to dataset
    def __getitem__( self, val ):
        data = self.dataset.__getitem__( self._translate_slice( val ) )
        return data

    def __setitem__( self, val, item ):
        self.dataset.__setitem__( self._translate_slice( val ), item )

    def __len__( self ):
        return self.get_num_samples()

    # Static create method for labeled timeseries
    @staticmethod
    def create( group, nsamp, labels, rate, name, dtype = 'float32', overwrite = False ):
        if isinstance( group, H5EEGGroup ): group = group.group
        num_ch = len( labels )
        shape = ( nsamp, num_ch )

        # Make sure it is okay to overwrite the data if necessary (and delete it)
        if not _okay_to_write( group, name, overwrite = overwrite ):
            return LabeledTimeseries( group[ name ] )

        dset = group.create_dataset( name, shape = shape, dtype = dtype )
        lts_dset = LabeledTimeseries( dset )
        lts_dset.set_labels( labels )
        lts_dset.set_rate( rate )
        return lts_dset

    # These functions convert time in seconds to samples.
    # It assumes that if time is specified as a float, it refers to seconds,
    # and if time is specified as an int, it refers to samples (no conversion).
    # This function probably goes against several python coding standards, but it
    # is provided here for convenience; if you don't like it, don't use it!
    def as_samples( self, time ):
        if isinstance( time, float ): return self.sec_to_samp( time )
        return time
    def as_seconds( self, time ):
        if isinstance( time, int ): return self.samp_to_sec( time )
        return time

class H5EEGAuxDataset( LabeledTimeseries, object ):
    """
    Auxiliary datasets contain non-electrophysiological data that corresponds
    to an eeg dataset (eg. microphone or accelerometer data).  This data is saved
    as integers, so there are no conversion factors.  Use LabeledTimeseries for
    floating point auxiliary datasets.  This currently exists mostly to set a 
    default name for the auxiliary dataset.
    """

    # Identifiers
    NAME = u'aux'

    # Constructor just passes dataset to superclass
    def __init__( self, dataset ):
        super( H5EEGAuxDataset, self ).__init__( dataset )

    # Static creation method
    @staticmethod
    def create( group, nsamp, labels, rate, name = NAME, overwrite = False ):
        """
        Create a dataset to hold auxiliary data

        Keyword Arguments:
        group -- The parent h5py.Group/H5EEGGroup to create the dataset in
        nsamp -- The number of samples in the dataset
        labels -- The names of the channels in the dataset
        rate -- Sampling rate in samples per second
        name -- The name of the dataset (default: 'aux')
        overwrite -- If a dataset named <name> already exists, use overwrite to 
            force the creation.  (default: False)

        Important implementation details:
        Shape of dataset = ( nsamp, nchannels )
        nchannels = len( labels )

        If name is not set to the default value, the default name will still be
        created in the dataset, as a soft-link that points to the named dataset.
        This functionality is actually preferred for more flexible data 
        representations such as grp.aux() pointing to ['raw'] or ['processed'] data.
        Whatever the 'aux' softlink *was* previously pointing to, it will now
        point to this newly created dataset.
        """
        # Create the dataset
        lts = LabeledTimeseries.create( group, nsamp, labels, rate, name, 
            dtype = 'int16', overwrite = overwrite )
        h5eegaux_dset = H5EEGAuxDataset( lts )

        # Create a softlink if necessary
        if name != H5EEGAuxDataset.NAME:
            group[ H5EEGAuxDataset.NAME ] = h5eegaux_dset.dataset

        return h5eegaux_dset

class H5EEGDataset( LabeledTimeseries, object ):
    """
    H5EEGDatasets contain electrophysiological data from a series of labeled 
    channels at a specific sampling rate.  Internally, data is currently stored
    as integers with offsets/gains for data compression's sake.  Data accessed
    using the __getitem__ interface (eg. H5EEGDataset_obj[t,ch]) is automatically 
    converted to uV by subtracting offset and multiplying the gain, so long as
    the decode_uv property is set to true.  Otherwise, the default behavior is to
    not scale the data.  Setting decode_uv to true drastically increases data 
    access time, and should only be performed if that actually matters to you.
    """
    
    # Identifiers
    NAME = u'eeg'
    OFFSETS = u'offsets'
    GAINS = u'gains'

    # Spatial filtering
    _spfilt = None
    _decode_uv = False

    # Constructor needs a dataset to encapsulate
    def __init__( self, dataset ):
        super( H5EEGDataset, self ).__init__( dataset )

    # Getters and setters
    def get_offsets( self ):
        """
        Get a list of channel DC offsets. (RAW - OFFSET) * GAIN = uV
        Returns None if they haven't been set.
        """
        if self.OFFSETS in self.dataset.attrs.keys():
            return np.squeeze( self.dataset.attrs[ self.OFFSETS ] )
        return np.zeros( len( self.get_labels() ) )

    def set_offsets( self, offsets ):
        """ Set the channel DC offsets. (list of integer offsets) """
        self.dataset.attrs[ self.OFFSETS ] = np.squeeze( offsets );

    def get_gains( self ):
        """
        Get a list of channel gains. (RAW - OFFSET) * GAIN = uV
        Returns None if they haven't been set.
        """
        if self.GAINS in self.dataset.attrs.keys():
            return np.squeeze( self.dataset.attrs[ self.GAINS ] )
        return np.ones( len( self.get_labels() ) )

    def set_gains( self, gains ):
        """ Set channel gains (list of floating point values) """
        self.dataset.attrs[ self.GAINS ] = np.squeeze( gains );

    @property
    def decode_uv( self ):
        return self._decode_uv

    @decode_uv.setter
    def decode_uv( self, value ):
        self._decode_uv = value

    def auto_CAR( self, exclude_ch = [] ):
        """
        This method will attempt to generate a Common Average Reference (CAR)
        spatial filter matrix for this object, based on the label names.

        Note: This method relies on the spfilt.py file packaged with ecogpy.

        Keyword Arguments:
            exclude_ch -- List of channels to exclude from CAR
        """
        try: 
            from spfilt import generate_CAR
            self._spfilt = generate_CAR( self.get_labels(), exclude_ch )
        except ImportError:
            raise Warning( "Failed to autogenerate CAR filter." )
            self._spfilt = None

    def set_spfilt( self, spfilt = None, override_square = False ):
        """ 
        Define a spatial filtering matrix that is applied
        during data access.  Must be square and have the shape
        (n,n) where n == len( self.get_labels() ).
        """
        # Manual spatial-filtering
        if isinstance( spfilt, np.ndarray ):
            if len( list( set( spfilt.shape ) ) ) != 1 and not override_square:
                raise ValueError( "spfilt should be a square matrix" )
            if spfilt.shape[0] != len( self.get_labels() ):
                raise ValueError( "spfilt is the wrong shape" )
            self._spfilt = spfilt[ :, : ]

        # Disable spatial filtering
        elif spfilt is None: self._spfilt = None

    # Data Access
    def __getitem__( self, val ):

        # Regardless of query, first grab all chans
        t_slice, ch_slice = self._translate_slice( val )
        slice_allchans = ( t_slice, slice( None ) )
        data = super( H5EEGDataset, self ).__getitem__( slice_allchans )

        # Transform the data to uV from AD units
        if self._decode_uv:
            data = self._transform( slice_allchans, data, self._ad_to_uv )
        
        # Apply spatial filtering if we want to do so.
        if self._spfilt is not None: data = np.dot( data, self._spfilt )

        # Only return the requested channels
        if( len( data.shape ) == 1 ):
            data = data[ ch_slice ]
        else: data = data[ :, ch_slice ]
        return data

    def __setitem__( self, val, data ):
        data = self._transform( val, data, self._uv_to_ad )
        super( H5EEGDataset, self ).__setitem__( val, data )

    # Define the transformation from A/D units to uV
    def _ad_to_uv( self, a, offset, gain ):
        """ Convert A/D units to uV """
        return np.multiply( np.subtract( a, offset ), gain )

    # Define the transform from uV to A/D units
    def _uv_to_ad( self, a, offset, gain ):
        """ Convert uV to A/D units """
        return np.add( np.divide( a, gain ), offset ).astype( self.dataset.dtype )

    # Transform a data slice to AD/uV
    def _transform( self, val, data, fn ):
        """
        Transform 'data' from slice 'val' by fn.  Internally, used to transform
        data from A/D units to uV and vice-versa.  'fn' is _ad_to_uv or _uv_to_ad
        """

        # Translate the slice into readable indices and acquire offsets/gains.
        val = self._translate_slice( val )
        offset = self.get_offsets()[ val[1] ]
        gain = self.get_gains()[ val[1] ]

        # Perform the transformation differently if input is an array
        if not isinstance( data, np.ndarray ): return fn( float( data ), offset, gain )
        elif data.ndim < 2: return fn( data.astype( 'float64' ), offset, gain )
        else: return np.apply_along_axis( fn, 1, data.astype( 'float64' ), offset, gain )

    # Static Creation Method
    @staticmethod
    def create( group, nsamp, labels, offsets, gains, rate, name = NAME,
        bytes_per_sample = 2, overwrite = False ):
        """
        Create a dataset to hold electophysiology data

        Keyword Arguments:
        group -- The parent h5py.Group/H5EEGGroup to create the dataset in
        nsamp -- The number of samples in the dataset
        labels -- The names of the channels in the dataset
        offsets -- The offsets for each channel
        gains -- The gains for each channel
        rate -- The sampling rate in samples per second
        name -- The name of the dataset (default: 'eeg') NOTE BELOW!
        bytes_per_sample -- The number of bytes to use for each sample (default: 2)
        overwrite -- If a dataset named <name> already exists, use overwrite to 
            force the creation.  (default: False)

        Important implementation details:
        Signal in uV = ( signal - offsets ) * gains
        Shape of dataset = ( nsamp, nchannels )
        nchannels = len( labels ) = len( offsets ) = len( gains )

        If name is not set to the default value, the default name will still be
        created in the dataset, as a soft-link that points to the named dataset.
        This functionality is actually preferred for more flexible data 
        representations such as grp.eeg() pointing to ['raw'] or ['car'] data.
        Whatever the 'eeg' softlink *was* previously pointing to, it will now
        point to this newly created dataset.
        """
        if len( offsets ) != len( labels ) or len( gains ) != len( labels ):
            Warning( "Size of labels, offsets, and gains should be consistent." )

        # Create the dataset
        sigdtype = '<i%d' % bytes_per_sample
        lts = LabeledTimeseries.create( group, nsamp, labels, rate, name, 
            dtype = sigdtype, overwrite = overwrite )
        h5eeg_dset = H5EEGDataset( lts )
        h5eeg_dset.set_offsets( offsets )
        h5eeg_dset.set_gains( gains )

        # Create a softlink if necessary
        if name != H5EEGDataset.NAME:
            group[ H5EEGDataset.NAME ] = h5eeg_dset.dataset

        return h5eeg_dset

class H5EEGEvents( object ):
    """
    H5EEGEvents provides an interface for marking sections of the data and 
    intelligently querying these markers.

    H5EEGEvents Interfaces with an internal event dataset as a database.  
    Upon instantiation, the class will extract events from the dataset and store
    them in memory. Events can be added or removed from this temporary copy in 
    memory.  Upon calling the 'write' method, the internal copy will be written 
    to the h5 dataset, deleting the old dataset and creating a new dataset.  
    Know that this will invalidate any other pointers to the old dataset.  
    Generally, work with this class as a read-only collection of events and only
    write the data to the file when you know no other references exist.
    """
    NAME = 'events'
    ID = 'name'
    START = 'start_idx'
    DURATION = 'duration'

    # We can't store unicode in h5eeg files, so we'll maintain an interface
    _EVENT_DTYPE = np.dtype( [ ( ID, np.string_, 32 ), ( START, 'i' ), ( DURATION, 'i' ) ] )
    EVENT_DTYPE = np.dtype( [ ( ID, np.str_, 32 ), ( START, 'i' ), ( DURATION, 'i' ) ] )

    dataset = None
    events = None

    def __init__( self, dataset ):
        """ Interpret the input dataset as an H5EEGDataset """
        self.dataset = dataset
        self.events = self.dataset[:].astype( self.EVENT_DTYPE )

    # Returns an index array
    def query( self, query, exact = False, during = None, length = 1 ):
        """
        Query the event dataset and return an index array of sample indices that
        match the query.

        Keyword Arguments:
        query -- A string that is compared to the event name.  If query is a
            substring of the event name, this event matches the query.
        exact -- If exact is set to true, the query must EXACTLY match the event
            name.  (Default: False)
        during -- If `during` is set to an array of indices, the start_idx of an
            event must be an element of this array of indices in addition to
            the above rules in order to be considered a match.  (Default: None)
            If during is set to an integer, it is assumed to be an index from
            the events array.  NOTE: Using an index array *can* be expensive.
            Protip: The output index array of this method can be used as the
            `during` input to a future query.  This can be used to perform sub-
            event classification/searching.
        length -- Used to specify the length of flag events.  If an event
            has duration '0', it is considered to be a 'flag', and the value
            specified here is used as the length for that event.  (Default: 1)
            NOTE: Length can be specified as a negative number to retreive 
            indices leading up to the flag.

        Returns a list of sorted and unique sample indices (an index array)
        corresponding to the query.
        """
        # Assemble an index buffer
        idx_buffer = []
        event_indices = self.query_event_indices( query, exact = exact, during = during )
        for event in self.events[ event_indices ]:

            # Add the indices of this event to the index buffer.
            if event[ self.DURATION ] != 0: length = event[ self.DURATION ]
            start = event[ self.START ]
            lo = max( min( start, start + length ), 0 )
            hi = max( max( start, start + length ), 0 )
            idx_buffer = idx_buffer + list( range( lo, hi ) )
        
        # Return a sorted list of unique indices
        return sorted( list( set( idx_buffer ) ) )

    # Returns a list of event indices
    def query_event_indices( self, query, exact = False, during = None ):
        """
        Query the event dataset and return indices of events matching the query.

        Keyword Arguments:
        query -- A string that is compared to the event name.  If query is a
            substring of the event name, this event matches the query.
        exact -- If exact is set to true, the query must EXACTLY match the event
            name.  (Default: False)
        during -- If `during` is set to an array of indices, the start_idx of an
            event must be an element of this array of indices in addition to
            the above rules in order to be considered a match.
            If `during` is set to an integer, it is assumed to be an index from
            the events array.  NOTE: Using an index array *can* be expensive.
            If `during` is of an event, the start_idx must be within the 
            specified event. `during` can also be set to an array of events.
            (Default: None)

        Returns a list of indices into the event array.
        TODO: HEAVY OPTIMIZATION
        """
        matching_event_indices = []
        for idx, event in enumerate( self.events ):

            # Determine if the query is a match
            query_match = False
            if not exact and query in event[ self.ID ]: query_match = True
            elif exact and query == event[ self.ID ]: query_match = True
            if not query_match: continue

            # If this event does not start in the during index buffer, ignore it.
            if isinstance( during, list ):
                if event[ self.START ] not in during: continue

            # If this event does not start in the during event index, ignore it.
            elif isinstance( during, int ):
                startidx = self.events[ during ][ self.START ]
                endidx = startidx + self.events[ during ][ self.DURATION ]
                if event[ self.START ] < startidx: continue
                if event[ self.START ] > endidx: continue

            # If this event does not exist in the during event, ignore it.
            elif type( during ) == np.void and during.dtype == self.EVENT_DTYPE:
                startidx = during[ self.START ]
                endidx = startidx + during[ self.DURATION ]
                if event[ self.START ] < startidx: continue
                if event[ self.START ] > endidx: continue

            # If this event does not exist in any of the during events, ignore.
            elif type( during ) == np.ndarray and during.dtype == self.EVENT_DTYPE:
                in_an_event = False
                for during_event in during:
                    startidx = during_event[ self.START ]
                    endidx = startidx + during_event[ self.DURATION ]
                    if event[ self.START ] > startidx and event[ self.START ] < endidx:
                        in_an_event = True
                if not in_an_event: continue


            # Add this event to the matching event indices
            matching_event_indices.append( idx )

        return matching_event_indices

    # Returns events directly
    def query_events( self, query, exact = False, during = None, labeler = None ):
        """
        Query the event dataset and return indices of events matching the query.

        Keyword Arguments:
        query -- A string that is compared to the event name.  If query is a
            substring of the event name, this event matches the query.
        exact -- If exact is set to true, the query must EXACTLY match the event
            name.  (Default: False)
        during -- If `during` is set to an array of indices, the start_idx of an
            event must be an element of this array of indices in addition to
            the above rules in order to be considered a match.  (Default: None)
            If during is set to an integer, it is assumed to be an index from
            the events array.  NOTE: Using an index array *can* be expensive.
        labeler -- a lambda that renames events based on some other logic.
            Default: None -- if unspecified, the event names are unchanged

        Returns an ndarray of event objects
        """
        # TODO: Make interaction with raw events a bit easier.  Write a wrapper class!
        ret = self[ self.query_event_indices( query, exact, during ) ]

        # Relabel events as necessary
        if labeler is not None:
            for event in ret: 
                event[ self.ID ] = labeler( event[ self.ID ] )

        return ret;

    # Write events (overwrite) to the dataset 
    def write( self, events = None ):
        """
        Write events to the events dataset.  Internally, this deletes the current
        dataset and creates a new one populated only with the input events.

        Keyword arguments:
        events -- a one dimensional numpy ndarray with the dtype EVENT_DTYPE
            (Default: None -- Write the internal representation of the events)

        NOTE: Internally, this will delete the old event dataset and write a new
        one.  This means that any pointers to the old dataset will become invalid.
        Use caution when invoking this method.
        """
        # Overwrite the temporary events store
        if events is not None: 
            self.events = events

        # Acquire the parent group and current name of the dataset
        parent = self.dataset.parent
        name = self.dataset.name

        # Out with the old and in with the new.
        del parent[ name ]
        self.dataset = parent.create_dataset( name, 
            data = self.events.astype( self._EVENT_DTYPE ) )

    def add_event( self, name, start_idx, duration = 0 ):
        """
        Add a new event to this dataset.
        Note: This change is not saved to the file unless `write` is called.

        Keyword Arguments:
        name -- The name of the event.  Used for queries/identification.
        start_idx -- The sample index of this event.
        duration -- The length of this event.  (Default: 0)
            Note: If duration = 0, this event is considered a 'Flag' with no set
            duration.  Flags can still be queried with the query interface, but
            the duration returned can be specified at a later point.
        """
        new_event = np.array( [ ( name, start_idx, duration ) ], dtype = self.EVENT_DTYPE )
        self.events = np.concatenate( [ self.events, new_event ] )

    # Direct access
    def __getitem__( self, val ):
        return self.events.__getitem__( val )

    def __setitem__( self, val, item ):
        """ Note: Make sure to `write` any changes you make! """
        self.events.__setitem__( val, item )

    def __len__( self ):
        return len( self.events )

    # Static Creation Method
    @staticmethod
    def create( group, name = NAME, overwrite = False ):
        """ 
        Create an empty event dataset in the group

        Keyword Args:
        group -- Parent group in which to create this dataset
        name -- Specify a name (Default: 'events')
        overwrite -- Mandate overwriting if necessary (Default: False)
        """
        if isinstance( group, H5EEGGroup ): group = group.group
        if not _okay_to_write( group, name, overwrite ): 
            return H5EEGEvents( group[ name ] )

        # Create the dataset initialized to no events.
        dset = group.create_dataset( name, ( 0, ), 
            dtype = H5EEGEvents._EVENT_DTYPE )
        return H5EEGEvents( dset )

# Small utility function to handle deletion/overwriting of existing datasets.
def _okay_to_write( parent, name, overwrite = False, warn = True ):
    """ WARNING: This will delete parent[name] if overwrite is set to true! """
    if name in parent.keys():
        if overwrite: del parent[ name ]
        else:
            if warn: Warning( "Dataset: %s already exists.  Use overwrite to force." % name )
            return False
    return True

def test_plots( group ):
    """ Plots the test data from the test group. """
    import pylab as pl
    pl.figure(); pl.plot( group.eeg()[:,'sin'] ); pl.show()
    pl.figure(); pl.plot( group.eeg()[:,'cos'] ); pl.show()
    pl.figure(); pl.plot( group.aux()[:,'saw'] ); pl.show()
    pl.figure(); pl.plot( group.eeg()[:,:] ); pl.show()

    # Plot some data queries
    pl.figure()
    pl.plot( group.eeg()[ group.events().query( 'FIRST_HALF' ), 'sin' ] )
    pl.plot( group.eeg()[ group.events().query( 'SECOND_HALF' ), 'cos' ] )
    pl.show()

    # Test plotting and channel queries
    group.eeg().plot_event( group.events()[ 0 ], channels = 'sin' ); pl.show();
    group.eeg().plot_event( group.events()[ 1 ], exclude = 'sin' ); pl.show();

# Unit Testing
def test():
    """ Creates a test file with some simple waveforms and makes some plots. """
    import tempfile

    tmpdir = tempfile.mkdtemp()
    filename = os.path.join( tmpdir, 'testsine.hdf5' )
    testfile = H5EEGFile( filename )

    group = H5EEGGroup.create( testfile, subject = 'test' )

    # Set up data parameters
    eeg_labels = [ 'sin', 'cos' ]
    eeg_offsets = [ 10, -10 ]
    eeg_gains = [ 0.0002, 0.0002 ]
    sample_rate = 30000.0
    num_samp = int( sample_rate * 10.0 )
    
    # Create the dataset
    H5EEGDataset.create( group, num_samp, eeg_labels, eeg_offsets, eeg_gains, 
        rate = sample_rate, bytes_per_sample = 2, name = 'test_raw' )

    # Set the data in the dataset (applies gains/offsets automatically)
    # NOTE: Usually the raw integer data is set manually.  This is an uncommon use case.
    group.eeg()[:,'sin'] = np.sin( np.arange( num_samp ) / sample_rate )
    group.eeg()[:,'cos'] = np.cos( np.arange( num_samp ) / sample_rate )

    # Create an auxiliary dataset containing metadata
    H5EEGAuxDataset.create( group, num_samp, labels = ['saw'], 
        rate = sample_rate, name = 'test_aux' )

    # Create a sawtooth for the aux data
    group.aux()[:,'saw'] = np.arange( num_samp ) % int( sample_rate )

    # Create some events
    events = H5EEGEvents.create( group )
    events.add_event( 'FILE', 0, num_samp )
    events.add_event( 'FIRST_HALF', 0, num_samp / 2 )
    events.add_event( 'SECOND_HALF', num_samp / 2, num_samp / 2 )
    events.add_event( 'FLAG_1', num_samp / 4 )
    events.add_event( 'FLAG_2', ( num_samp * 3 ) / 4 )
    events.write()
    
    # Close the file
    testfile.close()

    # Open the file as an h5eeg file and plot some data
    group = H5EEGFile( filename ).group
    test_plots( group )

    # Run some simple queries on the data
    # Match any event with 'FLAG' in the name during the event 'FIRST_HALF'
    result = group.events()[ group.events().query_event_indices( 'FLAG', 
        during = group.events().query( 'FIRST_HALF' ) )[0] ]
    assert( result[ 'name' ] == 'FLAG_1' ) # FLAG_1 is in the first half.
    
    # Match any event with 'FLAG' in the name during the event 'SECOND_HALF'
    result = group.events()[ group.events().query_event_indices( 'FLAG', 
        during = group.events().query( 'SECOND_HALF' ) )[0] ]
    assert( result[ 'name' ] == 'FLAG_2' ) # FLAG_2 is in the second half.

    # Cleanup
    if os.path.isfile( filename ):
        os.remove( filename )
    os.rmdir( tmpdir )

    print( 'Tests Passed.' )

if __name__=='__main__':
    test()
