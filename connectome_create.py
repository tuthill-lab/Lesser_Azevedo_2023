import utils
import pandas as pd
import numpy as np
import datetime as dt
from zoneinfo import ZoneInfo
import calendar
import os

pre_to_mn_path = os.path.join('.','dfs_pre_to_mn')


if __name__ == '__main__':
    pass


ts = dt.datetime.timetuple(dt.datetime.now(tz=ZoneInfo("America/Los_Angeles")))
timestamp = dt.datetime(ts[0],ts[1],ts[2],ts[3],tzinfo=ZoneInfo("America/Los_Angeles"))

timestamp = dt.datetime(2024, 1, 17, 8, 10, 1, 179472, tzinfo=ZoneInfo("America/Los_Angeles")) # v840 for the paper
                            #   dt.datetime(2023, 5, 25, 8, 10, 1, 254304, tzinfo=dt.timezone.utc) # v 604 for the preprint

print('query timestamp: {} ({})'.format(timestamp,timestamp.timestamp()))

def get_timestamp():
    return timestamp

def get_yesterday():
    return dt.datetime(ts[0],ts[1],ts[2],ts[3],tzinfo=ZoneInfo("America/Los_Angeles")) - dt.timedelta(days=1)

def left_pre_to_mn_df(client):
    t1_mns_df = client.materialize.query_table('motor_neuron_table_v7',timestamp=timestamp)
    soma_table = client.materialize.query_table('neuron_somas_dec2022',timestamp=timestamp)
    neckns_df = client.materialize.query_table('neck_connective_tag_table_v1',timestamp=timestamp)
    all_sensory = get_unduplicated_sensory_axon_table(client)

    mn_index = mn_multi_index(t1_mns_df)
    pre_to_mn_df,mn_index = create_pre_to_mn_df(client,mn_index,soma_table,neckns_df,all_sensory,['L'])

    save_pre_to_mn_df(pre_to_mn_df)
    return pre_to_mn_df

def get_unduplicated_sensory_axon_table(client):
    sensory_axons = client.materialize.query_table('nerve_bundle_fibers_v0',timestamp=timestamp)
    # sort to put "unsure" cell type at the bottom, then select the unduplicated
    sensory_axons = sensory_axons.sort_values('cell_type',ascending=True).loc[~sensory_axons.duplicated(keep='first',subset=['pt_root_id'])]
    sensory_axons = sensory_axons.loc[sensory_axons.pt_root_id>0]

    hair_plates = client.materialize.query_table('hair_plate_table',timestamp=timestamp)
    all_sensory = pd.concat([sensory_axons,hair_plates],    axis=0,    join="outer",    ignore_index=False,    keys=None,    levels=None,    names=None,    verify_integrity=False,    copy=True,)
    all_sensory = all_sensory.loc[~all_sensory.duplicated(keep='first',subset=['pt_root_id'])]
    return all_sensory

def get_unduplicated_sensory_axon_table_now(client):
    sensory_axons = client.materialize.query_table('nerve_bundle_fibers_v0')
    # sort to put "unsure" cell type at the bottom, then select the unduplicated
    sensory_axons = sensory_axons.sort_values('cell_type',ascending=True).loc[~sensory_axons.duplicated(keep='first',subset=['pt_root_id'])]
    sensory_axons = sensory_axons.loc[sensory_axons.pt_root_id>0]

    hair_plates = client.materialize.query_table('hair_plate_table')
    all_sensory = pd.concat([sensory_axons,hair_plates],    axis=0,    join="outer",    ignore_index=False,    keys=None,    levels=None,    names=None,    verify_integrity=False,    copy=True,)
    all_sensory = all_sensory.loc[~all_sensory.duplicated(keep='first',subset=['pt_root_id'])]
    return all_sensory

def get_live_sensory_axon_table(client):
    # sensory_axons = client.materialize.query_table('nerve_bundle_fibers_v0',timestamp=timestamp)
    sensory_axons = client.materialize.live_live_query('nerve_bundle_fibers_v0',timestamp='now')
    # sort to put "unsure" cell type at the bottom, then select the unduplicated
    sensory_axons = sensory_axons.sort_values('cell_type',ascending=True).loc[~sensory_axons.duplicated(keep='first',subset=['pt_root_id'])]
    sensory_axons = sensory_axons.loc[sensory_axons.pt_root_id>0]

    # hair_plates = client.materialize.query_table('hair_plate_table',timestamp=timestamp)
    hair_plates = client.materialize.live_live_query('hair_plate_table',timestamp='now')
    all_sensory = pd.concat([sensory_axons,hair_plates],    axis=0,    join="outer",    ignore_index=False,    keys=None,    levels=None,    names=None,    verify_integrity=False,    copy=True,)
    # all_sensory = all_sensory[['pt_root_id','classification_system','cell_type']]
    # all_sensory = all_sensory.rename({'pt_root_id':'segID'},axis=1)
    all_sensory = all_sensory.loc[~all_sensory.duplicated(keep='first',subset=['pt_root_id'])]
    return all_sensory

def load_or_create_pre_to_mn_df(client=None,from_pickle=True):
    if not from_pickle and client is None:
        raise ValueError('If from_pickle is False, client= must be supplied')
    name = 'pre_to_mn_df'
    today = dt.date.today()
    d1 = today.strftime("%Y%m%d")
    fn = pre_to_mn_path + os.sep + name + '_' + d1 + '.pkl'
    try:
        pre_to_mn_df = pd.read_pickle(fn)
    except FileNotFoundError:
        print('No file name: {}'.format(fn))
        if client is not None:
            print('File not found, creating left pre_to_mn_df')
            return left_pre_to_mn_df(client)
        else:
            raise FileNotFoundError()
    except BaseException as err:
        print(f"Unexpected {err}, {type(err)}")
        raise err
    print('Found pickle file {}'.format(fn))
    return pre_to_mn_df

def load_pre_to_mn_df(ext=''):
    name = 'pre_to_mn_df'
    if ext != '':
        name = 'pre_to_mn_df' + '_' + ext
    today = dt.date.today()
    d1 = today.strftime("%Y%m%d")
    fn = pre_to_mn_path + os.sep + name + '_' + d1 + '.pkl'
    try:
        pre_to_mn_df = pd.read_pickle(fn)    
    except FileNotFoundError:
        print('No pickle found, trying to load old {}'.format(ext))
        pre_to_mn_df = load_old_pre_to_mn_df(ext=ext)
    print('Found pickle file {}'.format(fn))
    return pre_to_mn_df 


def load_old_pre_to_mn_df(ext=''):
    name = 'pre_to_mn_df'
    if ext != '':
        name = 'pre_to_mn_df' + '_' + ext
    today = dt.date.today()
    b=0
    fnfe = False
    while b<10:
        b=b+1
        yday = today - dt.timedelta(days=b)
        d1 = yday.strftime("%Y%m%d")
        fn = pre_to_mn_path + os.sep + name + '_' + d1 + '.pkl'
        try:
            pre_to_mn_df = pd.read_pickle(fn)
            fnfe=False
            break
        except FileNotFoundError:
            fnfe=True
    if fnfe:
        print('No pickle found, trying to load old {}'.format(ext))
        raise(FileNotFoundError)
    else:        
        print('Found pickle file {}'.format(fn))
        return pre_to_mn_df 


def load_pre_to_df(ext=''):
    name = 'pre_to_df'
    if ext != '':
        name = 'pre_to_df' + '_' + ext
    today = dt.date.today()
    d1 = today.strftime("%Y%m%d")
    fn = './dfs_pre_to_/' + name + '_' + d1 + '.pkl'
    try:
        pre_to_df = pd.read_pickle(fn)    
    except FileNotFoundError:
        print('No pickle found, trying to load old {}'.format(ext))
        pre_to_df,fn = load_old_pre_to_df(ext=ext)
    print('Found pickle file {}'.format(fn))
    return pre_to_df 


def load_old_pre_to_df(ext=''):
    name = 'pre_to_df'
    if ext != '':
        name = 'pre_to_df' + '_' + ext
    today = dt.date.today()
    b=0
    fnfe = False
    while b<10:
        b=b+1
        yday = today - dt.timedelta(days=b)
        d1 = yday.strftime("%Y%m%d")
        fn = './dfs_pre_to_/' + name + '_' + d1 + '.pkl'
        try:
            pre_to_df = pd.read_pickle(fn)
            fnfe=False
            break
        except FileNotFoundError:
            fnfe=True
    if fnfe:
        print('No pickle found')
        raise(FileNotFoundError)
    else:        
        print('Found pickle file {}'.format(fn))
        return pre_to_df,fn 

def load_pre_of_df(ext=''):
    name = 'pre_of_df'
    if ext != '':
        name = 'pre_of_df' + '_' + ext
    today = dt.date.today()
    d1 = today.strftime("%Y%m%d")
    fn = './dfs_pre_of_/' + name + '_' + d1 + '.pkl'
    try:
        pre_to_df = pd.read_pickle(fn)    
    except FileNotFoundError:
        print('No pickle found, trying to load old {}'.format(ext))
        pre_to_df,fn = load_old_pre_of_df(ext=ext)
    print('Found pickle file {}'.format(fn))
    return pre_to_df 

def load_old_pre_of_df(ext=''):
    name = 'pre_of_df'
    if ext != '':
        name = 'pre_of_df' + '_' + ext
    today = dt.date.today()
    b=0
    fnfe = False
    while b<10:
        b=b+1
        yday = today - dt.timedelta(days=b)
        d1 = yday.strftime("%Y%m%d")
        fn = './dfs_pre_of_/' + name + '_' + d1 + '.pkl'
        try:
            pre_to_df = pd.read_pickle(fn)
            fnfe=False
            break
        except FileNotFoundError:
            fnfe=True
    if fnfe:
        print('No pickle found')
        raise(FileNotFoundError)
    else:        
        print('Found pickle file {}'.format(fn))
        return pre_to_df,fn 

def save_pre_to_mn_df(temp_df,ext=''):
    name = 'pre_to_mn_df'
    if ext != '':
        name = 'pre_to_mn_df' + '_' + ext
    today = dt.date.today()
    d1 = today.strftime("%Y%m%d")
    fn = pre_to_mn_path + os.sep + name + '_' + d1 + '.pkl'
    temp_df.to_pickle(fn)
    print(fn)
    print(temp_df.shape)
    
def save_pre_to_df(temp_df,ext=''):
    name = 'pre_to_df'
    if ext != '':
        name = 'pre_to_df' + '_' + ext
    today = dt.date.today()
    d1 = today.strftime("%Y%m%d")
    fn = './dfs_pre_to_/' + name + '_' + d1 + '.pkl'
    temp_df.to_pickle(fn)
    print(fn)
    print(temp_df.shape)

def save_pre_of_df(temp_df,ext=''):
    name = 'pre_of_df'
    if ext != '':
        name = 'pre_of_df' + '_' + ext
    today = dt.date.today()
    d1 = today.strftime("%Y%m%d")
    fn = './dfs_pre_of_/' + name + '_' + d1 + '.pkl'
    temp_df.to_pickle(fn)
    print(fn)
    print(temp_df.shape)

def mn_multi_index(mn_df):
    # Input: the motor neuron table dataframe
    # Output: a pandas multi index object

    side = []
    nerve = []
    segment = []
    function = []
    muscle = []
    rank = []
    segID = []

    # Iterate down the rows and find the segment, muscle, function rank from the cell_type name in the data frame
    for index, row in mn_df.iterrows():
        clss = row.classification_system
        nerve = nerve + [utils.xnerve(clss)]
        if 'R' in clss:
            side = side + ['R']
        elif 'L' in clss:
            side = side + ['L']
        
        ct = row.cell_type
        
        seg,rnk = utils.xsegment(ct)
        segment = segment+[seg]
        rank = rank+[rnk]
        
        mscl = utils.xmuscle(ct)
        muscle = muscle+[mscl]
        
        fcn = utils.xfunction(ct)    
        function = function+[fcn];

        segID = segID+[row.pt_root_id]
        
    # Standard way of creating a multiindex object
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html
    arrays = [
        side,
        nerve,
        segment,
        function,
        muscle,
        rank,
        segID,
             ]
    mn_tuples = list(zip(*arrays))
    mn_index = pd.MultiIndex.from_tuples(mn_tuples, names=['side','nerve',"segment", 'function','muscle','rank','segID'])
    return mn_index


def create_pre_to_mn_df(client,mn_index,soma_table,neckns_df,all_sensory,sides=['L']):
    # Iterates down the multiindex of motor neurons, selecting similar motor neuron segID to query for presynaptic partners.
    # Inputs: client, mn_index and optional side
    # returns: 
    if sides is None:
        sides = mn_index.get_level_values('side').to_list()
    nerves = sorted(mn_index.get_level_values('nerve').unique())
    segments = sorted(mn_index.get_level_values('segment').unique())
    fcns = sorted(mn_index.get_level_values('function').unique())

    print(sides)
    print(nerves)
    print(segments)
    print(fcns)

    mn_segIDs = mn_index.to_frame()
    mn_segIDs.index = mn_segIDs.segID

    # Initialize for control loop
    pre_to_mn_df = None

    # timestamp = dt.datetime.utcnow()
    print(timestamp)

    for sd in sides: 
        mnmi_side = utils.multiindex_include(mn_index,[sd])
        for nrv in nerves: # ['Leg']:
            mnmi_side_nrv = utils.multiindex_include(mnmi_side,[nrv])
            segs = mnmi_side_nrv.get_level_values('segment').to_list()
            segs = sorted([*{*segs}])
            for sg in segs:
                mnmi_side_nrv_seg = utils.multiindex_include(mnmi_side_nrv,[sg])
                segfcns = mnmi_side_nrv_seg.get_level_values('function').to_list()
                segfcns = sorted([*{*segfcns}])
                for fcn in segfcns:
                    mnmi_side_nrv_seg_fcn = utils.multiindex_include(mnmi_side_nrv_seg,[fcn])

                    print('({}, {}, {}, {})'.format(sd,nrv,sg,fcn))

                    # get the resulting segIDs
                    select_mn_segIDs = mnmi_side_nrv_seg_fcn.get_level_values('segID').to_list()

                    # Query the synapse table
                    mn_inputs_df = client.materialize.synapse_query(post_ids = select_mn_segIDs,timestamp=timestamp) # Takes list
                    mn_inputs_df['motor'] = mn_inputs_df.pre_pt_root_id.isin(mn_segIDs.segID)
                    mn_inputs_df['has_soma'] = mn_inputs_df.pre_pt_root_id.isin(soma_table.pt_root_id)
                    mn_inputs_df['sensory'] = mn_inputs_df.pre_pt_root_id.isin(all_sensory.pt_root_id)
                    mn_inputs_df['neck'] = mn_inputs_df.pre_pt_root_id.isin(neckns_df.pt_root_id)

                    # if (mn_inputs_df.pre_pt_root_id==648518346489965495).any():
                    #     print('HERE IT IS, FROM THE SYNAPSE QUERY!')
                    #     return mn_inputs_df, mn_index
                    print(mn_inputs_df.shape)

                    mn_inputs_df = group_and_count_inputs(mn_inputs_df,3)
                    partner_df = create_pre_post_df(mn_inputs_df,mnmi_side_nrv_seg_fcn)

                    if pre_to_mn_df is None:
                        pre_to_mn_df = partner_df
                    else:
                        # preserve the order of premotor neurons from most connected down
                        curidx = pre_to_mn_df.index.to_list()
                        newidx = partner_df.index.to_list()
                        for segid in newidx:
                            if segid not in curidx:
                                curidx.append(segid)
                        pre_to_mn_df = pd.concat([pre_to_mn_df,partner_df],axis=1,join='outer').reindex(index=curidx).fillna(value=0,downcast='infer')

    mn_index = pre_to_mn_df.columns           
    return pre_to_mn_df, mn_index



    # Given a dataframe of inputs, df, and a lower bound on the number of connections to keep
# Get a dataframe with counts of pre/post connections .
def group_and_count_inputs(df, thresh = 5):

    # count the number of synapses between pairs of pre and post synaptic inputs
    syn_in_conn=df.groupby(['pre_pt_root_id','post_pt_root_id']).transform(len)['id']
    # save this result in a new column and reorder the index
    df['syn_in_conn']=syn_in_conn
    df = df[['id', 'pre_pt_root_id','post_pt_root_id','score','syn_in_conn','motor','has_soma','sensory','neck']].sort_values('syn_in_conn', ascending=False).reset_index()

    # Filter out small synapses between pairs of neurons and now print the shape
    df = df[df['syn_in_conn']>=thresh]
    # print(df.shape)
    return df

# Given a data frame of pre/post pairs, with a column  grouped connections 
def create_pre_post_df(mn_inputs_df,mi):
    # Find the unique premotor neurons
    pre_mn_segIDs, idx = np.unique(mn_inputs_df['pre_pt_root_id'].to_list(),return_index=True)

    # Find out if they are motor neurons
    motor = mn_inputs_df['motor'][idx]

    # Find out if they have somas
    has_soma = mn_inputs_df['has_soma'][idx]
 
    # Find out if they are sensory
    sensory = mn_inputs_df['sensory'][idx]

    # Find out if they go through the neck connective
    neck = mn_inputs_df['neck'][idx]

    # Placeholder to find out if a neuron is local to T1
    local = neck.copy()
    local.loc[:] = False

    # Create multi index out of key info
    arrays = [
        pre_mn_segIDs,
        motor,
        has_soma,
        sensory,
        neck,
        local
            ]
    pmn_tuples = list(zip(*arrays))
    # pmn_index = pd.MultiIndex.from_tuples(pmn_tuples, names=['segID','motor,'has_soma','sensory','descending','ascending','intersegmental','local'])
    pmn_index = pd.MultiIndex.from_tuples(pmn_tuples, names=['segID','motor','has_soma','sensory','neck','local'])

    # Create a data frame with premotor neurons along the rows (index) and motor neurons along the columns
    if isinstance(mi,int):
        mi = [mi]
        
    dfdata = np.zeros([len(pre_mn_segIDs),len(mi)],dtype=int)
    mn_df = pd.DataFrame(dfdata,index=pmn_index, columns=mi)

    # Simplify the column indices
    mn_df.columns = mn_df.columns.get_level_values('segID')
    for index, row in mn_inputs_df.iterrows():
        mn_df.at[row.pre_pt_root_id,row.post_pt_root_id] = row.syn_in_conn

    # put the multiindex back
    mn_df.columns = mi
    return mn_df

def ordered_list_of_premotor_segIDs(client,segID,thresh = 5):
    a = [segID]
    arrays = [
        a]
    segID_tuples = list(zip(*arrays))
    segID_mi = pd.MultiIndex.from_tuples(segID_tuples, names=['segID'])
    mn_inputs_df = client.materialize.synapse_query(post_ids = segID_mi.get_level_values('segID').to_list())
    mn_inputs_df = group_and_count_inputs(mn_inputs_df, thresh = 5)
    partner_df = create_pre_post_df(mn_inputs_df,segID_mi)

def order_index_of_motor_neurons(multidx):
    order = []
    print('Implement the function')
    return order,multidx

def get_input_matrix(sids=[],client=None,thresh=3):
    # get all the synapses, then cross tab at the end
    # Peel off the index of the pre_to_mn list, which has info on premotor type and segIDs

    cnt = 0
    xi = 0
    delta = 16
    xf = delta

    pre_df = None

    timestamp = get_timestamp()

    # Iterate over the premotor neurons in chunks

    def remake_df(a):
        a_ = pd.DataFrame(a.values)
        a_.columns = a.columns
        return a_

    while xf<=len(sids):
        sids_ = sids[xi:xf]
        presyn_df = client.materialize.synapse_query(pre_ids = sids_,timestamp=timestamp) # Takes list
        presyn_df = remake_df(presyn_df)
        
        print(presyn_df.shape)

        if pre_df is None:
            pre_df = presyn_df
        else:
            pre_df = pd.concat([pre_df,presyn_df],axis=0,join='outer')
        
        cnt=cnt+1
        if cnt % 4 == 0:
            print('{} of {} rows complete, idx = {}'.format(xf,len(sids),sids_[-1]))
            print(pre_df.shape)
        xi=xf
        if xf+delta < len(sids):
            xf=xf+delta
        elif xf==len(sids):
            break
        else:
            xf = len(sids)

    # This takes more like 4 minutes.
    print(pre_df.shape)

    # OK, this is a crazy piece of code to calculate the crosstabulation
    # Currently, the dataframe is too large to perform crosstab. supposedly, pandas 1.4 solves this issue, but I'm having a hard time updating

    # first, for readability, just take the columns of interest
    df = pre_df.loc[:,['pre_pt_root_id','post_pt_root_id']]

    # Then use factorize, which encodes each occurance in a list, in this case of paired root_ids, pre and post
    ij,tups = pd.factorize(list(zip(*map(df.get,df))))

    # Then create a dictionary out of each occurance, storing the number of factorized codes
    result = dict(zip(tups, np.bincount(ij)))

    # Finally, turn the result into series, with the tuples of pt_root_ids as the multiindex
    tupseries = pd.Series(result)

    # Finally, threshold by the connection strength and perfom the unstack operation
    thresholdedtupseries = tupseries[tupseries>=thresh]
    prelim_pre_df = thresholdedtupseries.unstack(fill_value=0)

    prelim_pre_df = prelim_pre_df.T
    prelim_pre_df.shape

    # takes ~ 15 seconds,
    print(prelim_pre_df.shape)
    return prelim_pre_df

    # leftout = idx_df[~idx_df.segID.isin(pre_of_pre_df.columns)]
    # leftout # nothing should be left out

def get_output_matrix(sids=[],client=None,thresh=3):
    # get all the synapses, then cross tab at the end
    # Peel off the index of the pre_to_mn list, which has info on premotor type and segIDs

    cnt = 0
    xi = 0
    delta = 16
    xf = delta

    pre_df = None

    timestamp = get_timestamp()

    # Iterate over the premotor neurons in chunks

    def remake_df(a):
        a_ = pd.DataFrame(a.values)
        a_.columns = a.columns
        return a_

    while xf<=len(sids):
        sids_ = sids[xi:xf]
        presyn_df = client.materialize.synapse_query(post_ids = sids_,timestamp=timestamp) # Takes list
        presyn_df = remake_df(presyn_df)
        
        print(presyn_df.shape)

        if pre_df is None:
            pre_df = presyn_df
        else:
            pre_df = pd.concat([pre_df,presyn_df],axis=0,join='outer')
        
        cnt=cnt+1
        if cnt % 4 == 0:
            print('{} of {} rows complete, idx = {}'.format(xf,len(sids),sids_[-1]))
            print(pre_df.shape)
        xi=xf
        if xf+delta < len(sids):
            xf=xf+delta
        elif xf==len(sids):
            break
        else:
            xf = len(sids)

    # This takes more like 4 minutes.
    print(pre_df.shape)

    # OK, this is a crazy piece of code to calculate the crosstabulation
    # Currently, the dataframe is too large to perform crosstab. supposedly, pandas 1.4 solves this issue, but I'm having a hard time updating

    # first, for readability, just take the columns of interest
    df = pre_df.loc[:,['pre_pt_root_id','post_pt_root_id']]

    # Then use factorize, which encodes each occurance in a list, in this case of paired root_ids, pre and post
    ij,tups = pd.factorize(list(zip(*map(df.get,df))))

    # Then create a dictionary out of each occurance, storing the number of factorized codes
    result = dict(zip(tups, np.bincount(ij)))

    # Finally, turn the result into series, with the tuples of pt_root_ids as the multiindex
    tupseries = pd.Series(result)

    # Finally, threshold by the connection strength and perfom the unstack operation
    thresholdedtupseries = tupseries[tupseries>=thresh]
    prelim_pre_df = thresholdedtupseries.unstack(fill_value=0)

    prelim_pre_df = prelim_pre_df.T
    prelim_pre_df.shape

    # takes ~ 15 seconds,
    print(prelim_pre_df.shape)
    return prelim_pre_df

    # leftout = idx_df[~idx_df.segID.isin(pre_of_pre_df.columns)]
    # leftout # nothing should be left out