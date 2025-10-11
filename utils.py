"""Utility functions for RekordBox database operations."""

import datetime
from uuid import uuid4

import polars as pl
from pyrekordbox import Rekordbox6Database
from pyrekordbox.db6 import tables


def _validate_tag_not_exists(
    db: Rekordbox6Database, content_id: str, tag_id: str
) -> None:
    """Validates that a tag is not already associated with content.

    Parameters
    ----------
    db : Rekordbox6Database
        The RekordBox database instance to use for the operation.
    content_id : str
        The Content ID to check.
    tag_id : str
        The Tag ID to check.

    Raises
    ------
    ValueError : If the tag is already associated with the content.
    """
    existing = (
        db.query(tables.DjmdSongMyTag)
        .filter_by(ContentID=content_id, MyTagID=tag_id)
        .first()
    )
    if existing:
        raise ValueError(
            f"Tag with ID={tag_id} is already associated with content ID={content_id}"
        )


def add_tag_with_track(
    db: Rekordbox6Database, tag_id: str, content_id: str, track_no: int | None = None
) -> tables.DjmdSongMyTag:
    """Adds a tag to a content/track.

    Creates a new DjmdSongMyTag object corresponding to the given
    content and tag, linking them together in the database.

    Parameters
    ----------
    db : Rekordbox6Database
        The RekordBox database instance to use for the operation.
    tag_id : str
        The MyTag ID to add to the content. This is the ID of the tag
        in the djmdMyTag table.
    content_id : str
        The Content ID to add the tag to. This is the ID of the track
        in the djmdContent table.
    track_no : int, optional
        The track number for ordering multiple tags on a track.
        If not specified, the tag will be added to the end.

    Returns
    -------
    song_tag : DjmdSongMyTag
        The song tag object that was created linking the content and tag.

    Raises
    ------
    ValueError : If the tag is already associated with the content.
    ValueError : If the track number is less than 1 or too large.

    Examples
    --------
    Add a tag to a content item:

    >>> db = Rekordbox6Database()
    >>> cid = '217615930'  # Content ID
    >>> tid = '382831235'  # Tag ID
    >>> db.add_tag(db, tid, cid)
    <DjmdSongMyTag(...)>

    Add a tag with a specific track number:

    >>> new_song_tag = add_tag(db, tid, cid, track_no=1)
    >>> new_song_tag.TrackNo
    1
    """
    # Get the content and tag objects
    content = db.get_content(ID=content_id)
    tag = db.get_my_tag(ID=tag_id)

    cid = content.ID
    tid = tag.ID

    # Check if the tag is already associated with this content
    _validate_tag_not_exists(db, cid, tid)

    # Generate IDs and timestamp
    uuid = str(uuid4())
    id_ = str(uuid4())
    now = datetime.datetime.now()

    # Count existing tags for this content
    ntags = db.query(tables.DjmdSongMyTag).filter_by(ContentID=content.ID).count()

    if track_no is not None:
        insert_at_end = False
        track_no = int(track_no)
        if track_no < 1:
            raise ValueError("Track number must be greater than 0")
        if track_no > ntags + 1:
            raise ValueError(f"Track number too high, content has {ntags} tags")
    else:
        insert_at_end = True
        track_no = ntags + 1

    print(f"Adding tag with ID={tid} to content with ID={cid}")
    print(f"Content ID:  {cid}")
    print(f"Tag ID:      {tid}")
    print(f"ID:          {id_}")
    print(f"UUID:        {uuid}")
    print(f"TrackNo:     {track_no}")

    moved = list()
    if not insert_at_end:
        db.registry.disable_tracking()
        # Update track numbers higher than the removed track
        query = (
            db.query(tables.DjmdSongMyTag)
            .filter(
                tables.DjmdSongMyTag.ContentID == content.ID,
                tables.DjmdSongMyTag.TrackNo >= track_no,
            )
            .order_by(tables.DjmdSongMyTag.TrackNo)
        )
        for other_song_tag in query:
            other_song_tag.TrackNo += 1
            other_song_tag.updated_at = now
            moved.append(other_song_tag)
        db.registry.enable_tracking()

    # Add tag to content
    song_tag: tables.DjmdSongMyTag = tables.DjmdSongMyTag.create(
        ID=id_,
        MyTagID=str(tid),
        ContentID=str(cid),
        TrackNo=track_no,
        UUID=uuid,
        created_at=now,
        updated_at=now,
    )
    db.add(song_tag)
    if not insert_at_end:
        moved.append(song_tag)
        db.registry.on_move(moved)

    return song_tag


def add_tag(
    db: Rekordbox6Database, tag_id: str, content_id: str
) -> tables.DjmdSongMyTag:
    """Adds a tag to a content/track (simplified version).

    Creates a new DjmdSongMyTag object corresponding to the given
    content and tag, linking them together in the database.
    The tag is always added to the end of the content's tag list.

    Parameters
    ----------
    db : Rekordbox6Database
        The RekordBox database instance to use for the operation.
    tag_id : str
        The MyTag ID to add to the content. This is the ID of the tag
        in the djmdMyTag table.
    content_id : str
        The Content ID to add the tag to. This is the ID of the track
        in the djmdContent table.

    Returns
    -------
    song_tag : DjmdSongMyTag
        The song tag object that was created linking the content and tag.

    Raises
    ------
    ValueError : If the tag is already associated with the content.

    Examples
    --------
    Add a tag to a content item:

    >>> db = Rekordbox6Database()
    >>> cid = '217615930'  # Content ID (Lords of the underground)
    >>> tid = '382831235'  # Tag ID (TAG YEAR tag)
    >>> song_tag = add_tag(db, tid, cid)
    >>> # Don't forget to commit: db.commit()
    """
    # Get the content and tag objects
    content = db.get_content(ID=content_id)
    tag = db.get_my_tag(ID=tag_id)

    cid = content.ID
    tid = tag.ID

    # Check if the tag is already associated with this content
    _validate_tag_not_exists(db, cid, tid)

    # Generate IDs and timestamp
    uuid = str(uuid4())
    id_ = str(uuid4())
    now = datetime.datetime.now()

    # Count existing tags for this content to determine track number
    ntags = db.query(tables.DjmdSongMyTag).filter_by(ContentID=content.ID).count()
    track_no = ntags + 1

    print(f"Adding tag with ID={tid} to content with ID={cid}")
    print(f"Content ID:  {cid}")
    print(f"Tag ID:      {tid}")
    print(f"ID:          {id_}")
    print(f"UUID:        {uuid}")
    print(f"TrackNo:     {track_no}")

    # Add tag to content
    song_tag: tables.DjmdSongMyTag = tables.DjmdSongMyTag.create(
        ID=id_,
        MyTagID=str(tid),
        ContentID=str(cid),
        TrackNo=track_no,
        UUID=uuid,
        created_at=now,
        updated_at=now,
    )
    db.add(song_tag)

    return song_tag


def get_db_content(db: Rekordbox6Database) -> pl.DataFrame:
    """
    Returns subset of content from RekordBox master.db table.
    """
    schema = {
        "ContentID": pl.Utf8,
        "FolderPath": pl.Utf8,
        "Title": pl.Utf8,
        "ArtistID": pl.Utf8,
        "ArtistName": pl.Utf8,
        "GenreID": pl.Utf8,
        "GenreName": pl.Utf8,
        "BPM": pl.Int32,
        "DateCreated": pl.Utf8,
        "Length": pl.Int32,
        "MyTagNames": pl.List(pl.Utf8),
        "MyTagIDs": pl.List(pl.Utf8),
    }

    return pl.from_dicts(
        data=[
            {
                "ContentID": content.ID,
                "FolderPath": content.FolderPath,
                "Title": content.Title,
                "ArtistID": content.ArtistID,
                "ArtistName": content.ArtistName,
                "GenreID": content.GenreID,
                "GenreName": content.GenreName,
                "BPM": content.BPM,
                "DateCreated": content.DateCreated,
                "Length": content.Length,
                "MyTagNames": list(content.MyTagNames) if content.MyTagNames else [],
                "MyTagIDs": list(content.MyTagIDs) if content.MyTagIDs else [],
            }
            for content in db.get_content()
        ],
        schema=schema,
    )
