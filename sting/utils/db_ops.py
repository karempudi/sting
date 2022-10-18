
# Contains all the functions needed to 
# write to database of a live or simulated live
# experiment.

import sys
import sqlite3
from datetime import datetime
from pathlib import Path

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
                    segmentedpath TEXT, rawpath TEXT, barcodes INTEGER, locations BLOB, numchannels BLOB)""")
            elif db == 'track':
                cur.execute("""CREATE TABLE if not exists track
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, time TIMESTAMP, position INTEGER, timepoint INTEGER,
                    channelno INTEGER, beforebarcode INTEGER, afterbarcode INTEGER, location INTEGER)""")
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
        keys = ['position', 'timepoint', 'segmentedpath', 'rawpath', 'barcodes', 'locations', 'numchannels']
        for key in keys:
            if key not in event_data:
                event_data[key] = None
        con = None
        db_file = dir_name / Path('segment.db')
        try:
            con = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None) 
            cur = con.cursor()
            cur.execute("""INSERT into segment (time, position, timepoint, segmentedpath, rawpath, barcodes,
                        locations, numchannels) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                        (datetime.now(), event_data['position'], event_data['timepoint'], event_data['segmentedpath'],
                        event_data['rawpath'], event_data['barcodes'], event_data['locations'], event_data['numchannels'],)
                    )
        except Exception as e:
            sys.stderr.write(f"Error while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
 
    elif event_type == 'track':
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
            sys.stderr.write(f"Error while writing to table {event_type} -- {event_data}\n")
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
            sys.stderr.write(f"Error while writing to table {event_type} -- {event_data}\n")
            sys.stderr.flush()
        finally:
            if con:
                con.close()
 
 

def read_from_db(event_type, event_args):
    if event_type == 'acquire':
        pass
    elif event_type == 'segment':
        pass
    elif event_type == 'track':
        pass
    elif event_type == 'growth':
        pass
