
# Contains all the functions needed to 
# write to database of a live or simulated live
# experiment.

import sys
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def create_databases(dir_name, db_names):
    """
    Creates one database for each of the queues that you use in the 
    
    Arguments:
        dir_name: directory in which sqlite databases will be created
        db_names: for each name in the list of db_names, a new database
            is created with exactly one table with the same name as the 
            database and we write with the schema designed only in this file
            and nowhere else. If you want to change the table schema, do it only
            in this file
    """
    for db in db_names:
        sys.stdout.write(f"Creating {db} datatable ... \n")
        sys.stdout.flush()
        db_file = dir_name / Path(db + '.db')
        # if the file exists don't create new table
        con = None
        try:
            if not db_file.exists(): 
                con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
            else:
                # find table if not create
                sys.stdout.write(f"{db} database exists. Will append to tables \n")
                sys.stdout.flush()
                con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)

            cur = con.cursor()
            if db == 'acquire':
                cur.execute("""CREATE TABLE if not exists acquire 
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, position INTEGER, timepoint INTEGER)
                    """)
            elif db == 'segment':
                cur.execute("""CREATE TABLE if not exists segment
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, position INTEGER, timepoint INTEGER,
                    rawpath TEXT, barcodes INTEGER, barcodelocations TEXT, numchannels INTEGER, channellocations TEXT)""")
            elif db == 'track':
                cur.execute("""CREATE TABLE if not exists track
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, position INTEGER, timepoint INTEGER) """)
            elif db == 'growth':
                cur.execute("""CREATE TABLE if not exists growth
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, position INTEGER, timepoint INTEGER,
                    channelno INTEGER, beforebarcode INTEGER, afterbarcode INTEGER, location INTEGER)""")
        except Exception as e:
            sys.stderr.write(f"Exception {e} raised in database creation :( \n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()

def write_to_db(event_data, dir_name, event_type):
    if event_type == 'acquire':
        keys = ['position', 'timepoint']
        for key in keys:
            if key not in event_data:
                event_data[key] = None
        con = None
        db_file = dir_name / Path('acquire.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None) 
            cur = con.cursor()
            cur.execute("""INSERT into acquire (time, position, timepoint) VALUES (?, ?, ?)""", 
                        (datetime.now(), event_data['position'], event_data['timepoint'],))
        except Exception as e:
            sys.stderr.write(f"Error {e} while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
        
    elif event_type == 'segment':
        keys = ['position', 'time', 'total_channels', 'error']
        for key in keys:
            if key not in event_data:
                raise ValueError("Segmentation Database write received an invalid input ...")
        
        event_data['rawpath'] = dir_name / Path('Pos'+ str(event_data['position'])) / Path('phase') /Path('phase_' + str(event_data['time']).zfill(4)+ '.tiff')
        event_data['barcodes'] = len(event_data['barcode_locations'])  
    
        if event_data['error']:
            event_data['barcodes'] = 0
            event_data['barcode_locations'] = [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
            event_data['channel_locations_list'] = [-1.]

        #print(json.dumps(event_data['barcode_locations']))
        #print(str(event_data['rawpath']))
        #print(type(event_data['barcode_locations']))
        #print(json.dumps(event_data['barcode_locations']))
        
        con = None
        db_file = dir_name / Path('segment.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None) 
            cur = con.cursor()
            cur.execute("""INSERT into segment (time, position, timepoint, rawpath, barcodes,
                        barcodelocations, numchannels, channellocations) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                        (datetime.now(), event_data['position'], event_data['time'],
                        str(event_data['rawpath']), event_data['barcodes'], json.dumps(event_data['barcode_locations'], cls=NpEncoder), 
                        int(event_data['total_channels']), json.dumps(event_data['channel_locations_list']))
                    )
        except Exception as e:
            sys.stderr.write(f"Error {e} while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
 
    elif event_type == 'track':
        keys = ['position', 'time']
        for key in keys:
            if key not in event_data:
                event_data[key] = None
        con = None
        db_file = dir_name / Path('track.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None) 
            cur = con.cursor()
            cur.execute("""INSERT into track (time, position, timepoint) VALUES (?, ?, ?)""", 
                        (datetime.now(), event_data['position'], event_data['time'])
                    )
        except Exception as e:
            sys.stderr.write(f"Error {e} while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
 
    elif event_type == 'growth':
        keys = ['position', 'timepoint', 'channelno', 'beforebarcode', 'afterbarcode', 'location']
        for key in keys:
            if key not in event_data:
                event_data[key] = None
        con = None
        db_file = dir_name / Path('track.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None) 
            cur = con.cursor()
            cur.execute("""INSERT into segment (time, position, timepoint, channelno, beforebarcode, afterbarcode,
                        location) VALUES (?, ?, ?, ?, ?, ?, ?)""", 
                        (datetime.now(), event_data['position'], event_data['timepoint'], event_data['channelno'],
                        event_data['beforebarcode'], event_data['afterbarcode'], event_data['location'],)
                    )
        except Exception as e:
            sys.stderr.write(f"Error {e} while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
 
 

def read_from_db(event_type, dir_name, position=None, timepoint=None):
    if event_type == 'acquire':
        con = None
        position, timepoint = None, None
        db_file = dir_name / Path('acquire.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
            cur = con.cursor()
            cur.execute("SELECT position, timepoint FROM acquire ORDER BY id DESC LIMIT 1")

            position, timepoint = cur.fetchone()
        except Exception as e:
            sys.stdout.write(f"Error {e} while fetching from table {event_type} -- {dir_name}\n")
            sys.stdout.flush()

        finally:
            if con:
                con.close()
            return position, timepoint
            
    elif event_type == 'segment':
        con = None
        position, timepoint = None, None
        db_file = dir_name / Path('segment.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
            cur = con.cursor()
            cur.execute("SELECT position, timepoint FROM segment ORDER BY id DESC LIMIT 1")

            position, timepoint = cur.fetchone()

        except Exception as e:
            sys.stdout.write(f"Error {e} while fetching from table {event_type} -- {dir_name}\n")
            sys.stdout.flush()

        finally:
            if con:
                con.close()
            return position, timepoint

    elif event_type == 'track':
        con = None
        position, timepoint = None, None
        db_file = dir_name / Path('track.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)
            cur = con.cursor()
            cur.execute("SELECT position, timepoint FROM track ORDER BY id DESC LIMIT 1")

            position, timepoint = cur.fetchone()

        except Exception as e:
            sys.stdout.write(f"Error {e} while fetching from table {event_type} -- {dir_name}\n")
            sys.stdout.flush()

        finally:
            if con:
                con.close()
            return position, timepoint
    elif event_type == 'barcode_locations':
        con = None
        # position and timepoint are the key work args
        db_file = dir_name / Path('segment.db')
        data = None
        #print(f"Getting barcode locations form {db_file}, position: {position}, timepoint: {timepoint}")
        try:
            con = sqlite3.connect(db_file)
            cur = con.cursor()
            cur.execute("""SELECT barcodes, barcodelocations, numchannels, channellocations FROM segment WHERE (position=? AND timepoint=?)""", (position, timepoint))
            data = cur.fetchone()
            data = {'numbarcodes': int(data[0]),
                    'barcode_locations': json.loads(data[1]),
                    'numchannels': int(data[2]),
                    'channel_locations': json.loads(data[3])}   
        except Exception as e:
            sys.stdout.write(f"Error {e} while fetching from table segment: barcode_locations -- {dirname}\n")
            sys.stdout.flush()
        finally:
            if con:
                con.close()
            return data

    elif event_type == 'growth':
        pass
