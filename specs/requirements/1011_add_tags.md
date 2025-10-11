# Feature Requirement Document: add_tag function

I want to write back custom tags to RekordBox (DJ Software)using the PyRekordBox library.
The idea is to add multiple `DjmdSongMyTag` instances to the database, referencing existing `Djmdmytag` and `ContentID`. This is new functionality in the library, that I probably need to figure out myself, but there are other similar functions that I can look into to implement this. The library makes use of SQLAlchemy to map to the sqlite database of rekordbox. 

## Background Documentation

- [The github repo](https://github.com/dylanljones/pyrekordbox)
- [The documentation](https://pyrekordbox.readthedocs.io/)
- [Querying the Database:](https://pyrekordbox.readthedocs.io/en/stable/tutorial/db6.html#querying-the-database)
- [My github issue](https://github.com/dylanljones/pyrekordbox/issues/188)

## Dataset

I have queried the data and made an export of all the tags I currently have (name) and their corresponding ID.
This list of tags is exported at `data/unique_tags.csv, and looks something like:

```markdown
...
MyTagName, MyTagID
Afrobeat,989601273
After Hours,2355752193
Ambient,2484825285
Arabic,1817218090
Background,1783663050
Ballad,3711467963
Bass,681822852
Beats,2053683127
Belgium,2280727809
Blues,317314193
Boogie,2978164233
...
```

Similarly, I have a dataset of content in `data/content_sample.json` that contains the song info I have.

The idea is to use the `ContentID` from this dataset, and the `MyTagID` to add new records to a join table between `djmdContent` and `djmdMyTag`, called `djmdSongMyTag`.

## Data Models

### djmdContent

[The djmdcontent model](https://pyrekordbox.readthedocs.io/en/stable/formats/db6.html#djmdcontent): This table stores the main track data of Rekordbox. The table contains most information about each track in the collection. Some columns are linked to other tables by the corresponding ID.

#### djmdContent columns

Note; only important columns listed for this feature. 

- ID: The ID of the content, referenced as ContentID in other tables
- Title: The title of the track
- FileNameL: The long file name, This is the normal file name
- ... (many more)

The python data model is too big to show here, so we left this out. 

### djmdMyTag

[The djmdmytag model](https://pyrekordbox.readthedocs.io/en/stable/formats/db6.html#djmdmytag): This table stores the My-Tag data of Rekordbox. It does not store the tracks for which the My-Tag values are set. These are stored in the djmdSongMyTag table. The items in the table can either be a My-Tag section or an actual My-Tag value.

#### djmdMyTag columns

- ID: The ID of the My-Tag
- Seq: The number of the My-Tag entry, Used for sorting
- Name: The name of the My-Tag
- Attribute: The attributes of the My-Tag
- ParentID: The ID of the parent My-Tag section

```python
class DjmdMyTag(Base, StatsFull):
    """Table for storing My-Tags lists in the Rekordbox library.

    See Also
    --------
    :class:`DjmdSongMyTag`: Table for storing the My-Tag items.
    """

    __tablename__ = "djmdMyTag"

    ID: Mapped[str] = mapped_column(
        VARCHAR(255), ForeignKey("djmdMyTag.ParentID"), primary_key=True
    )
    """The ID (primary key) of the table entry."""
    Seq: Mapped[int] = mapped_column(Integer, default=None)
    """The sequence of the My-Tag list (for ordering)."""
    Name: Mapped[str] = mapped_column(VARCHAR(255), default=None)
    """The name of the My-Tag list."""
    Attribute: Mapped[int] = mapped_column(Integer, default=None)
    """The attribute of the My-Tag list."""
    ParentID: Mapped[str] = mapped_column(VARCHAR(255), ForeignKey("djmdMyTag.ID"), default=None)
    """The ID of the parent My-Tag list (:class:`DjmdMyTag`)."""

    MyTags = relationship("DjmdSongMyTag", back_populates="MyTag")
    """The My-Tag items (links to :class:`DjmdSongMyTag`)."""
    Children = relationship(
        "DjmdMyTag", foreign_keys=ParentID, backref=backref("Parent", remote_side=[ID])
    )
    """The child lists of the My-Tag list (links to :class:`DjmdMyTag`).
    Backrefs to the parent list via :attr:`Parent`.
    """

    def __repr__(self) -> str:
        s = f"{self.ID: <2} Name={self.Name}"
        return f"<{self.__class__.__name__}({s})>"
```

### djmdSongMyTag

[The djmysongmytag model](https://pyrekordbox.readthedocs.io/en/stable/formats/db6.html#djmdsongmytag): This table stores the My-tag values of tracks linked to in the djmdMyTag table.

djmdSongMyTag columns

- ID: The ID of the My-Tag value
- MyTagID: The ID of the My-Tag group containing the item, Links to ID in the djmdMyTag table
- ContentID: The corresponding track, Links to ID in the djmdContent table
- TrackNo: The number of the My-Tag for a track

```python
class DjmdSongMyTag(Base, StatsFull):
    """Table for storing My-Tag items in the Rekordbox library.

    See Also
    --------
    :class:`DjmdMyTag`: Table for storing My-Tag lists.
    """

    __tablename__ = "djmdSongMyTag"

    ID: Mapped[str] = mapped_column(VARCHAR(255), primary_key=True)
    """The ID (primary key) of the table entry."""
    MyTagID: Mapped[str] = mapped_column(VARCHAR(255), ForeignKey("djmdMyTag.ID"), default=None)
    """The ID of the My-Tag list (links to :class:`DjmdMyTag`)."""
    ContentID: Mapped[str] = mapped_column(VARCHAR(255), ForeignKey("djmdContent.ID"), default=None)
    """The ID of the content this item belongs to (:class:`DjmdContent`)."""
    TrackNo: Mapped[int] = mapped_column(Integer, default=None)
    """The track number of the My-Tag item (for ordering)."""

    MyTag = relationship("DjmdMyTag", back_populates="MyTags")
    """The My-Tag list this item belongs to (links to :class:`DjmdMyTag`)."""
    Content = relationship("DjmdContent", back_populates="MyTags")
    """The content this item belongs to (links to :class:`DjmdContent`)."""

    MyTagName = association_proxy("MyTag", "Name")
    """The name of the My-Tag item (:class:`DjmdMyTag`)."""
```

## Inspiration

The function below does something similar to what we'd like to do: associating a song ID with a Playlist ID, through a join table. In our case, we'd like to do the same for Song ID with Tag ID.

```python
def add_to_playlist(
        self, playlist: PlaylistLike, content: ContentLike, track_no: int = None
    ) -> tables.DjmdSongPlaylist:
        """Adds a track to a playlist.

        Creates a new :class:`DjmdSongPlaylist` object corresponding to the given
        content and adds it to the playlist.

        Parameters
        ----------
        playlist : DjmdPlaylist or int or str
            The playlist to add the track to. Can either be a :class:`DjmdPlaylist`
            object or a playlist ID.
        content : DjmdContent or int or str
            The content to add to the playlist. Can either be a :class:`DjmdContent`
            object or a content ID.
        track_no : int, optional
            The track number to add the content to. If not specified, the track
            will be added to the end of the playlist.

        Returns
        -------
        song: DjmdSongPlaylist
            The song playlist object that was created from the content.

        Raises
        ------
        ValueError : If the playlist is a folder or smart playlist.
        ValueError : If the track number is less than 1 or to large.

        Examples
        --------
        Add a track to the end of a playlist:

        >>> db = Rekordbox6Database()
        >>> cid = 12345  # Content ID
        >>> pid = 56789  # Playlist ID
        >>> db.add_to_playlist(pid, cid)
        <DjmdSongPlaylist(c803dfde-2236-4659-b3d7-e57221663375)>

        Add a track to the beginning of a playlist:

        >>> new_song = db.add_to_playlist(pid, cid, track_no=1)
        >>> new_song.TrackNo
        1
        """
        plist: DjmdPlaylist
        cont: DjmdContent
        if isinstance(playlist, (int, str)):
            plist = self.get_playlist(ID=playlist)
        else:
            plist = playlist

        if isinstance(content, (int, str)):
            cont = self.get_content(ID=content)
        else:
            cont = content

        # Check playlist attribute (can't be folder or smart playlist)
        if plist.Attribute != 0:
            raise ValueError("Playlist must be a normal playlist")

        uuid = str(uuid4())
        id_ = str(uuid4())
        now = datetime.datetime.now()
        nsongs = self.query(tables.DjmdSongPlaylist).filter_by(PlaylistID=plist.ID).count()
        if track_no is not None:
            insert_at_end = False
            track_no = int(track_no)
            if track_no < 1:
                raise ValueError("Track number must be greater than 0")
            if track_no > nsongs + 1:
                raise ValueError(f"Track number too high, parent contains {nsongs} items")
        else:
            insert_at_end = True
            track_no = nsongs + 1

        cid = cont.ID
        pid = plist.ID

        logger.info("Adding content with ID=%s to playlist with ID=%s:", cid, pid)
        logger.debug("Content ID:  %s", cid)
        logger.debug("Playlist ID: %s", pid)
        logger.debug("ID:          %s", id_)
        logger.debug("UUID:        %s", uuid)
        logger.debug("TrackNo:     %s", track_no)

        moved = list()
        if not insert_at_end:
            self.registry.disable_tracking()
            # Update track numbers higher than the removed track
            query = (
                self.query(tables.DjmdSongPlaylist)
                .filter(
                    tables.DjmdSongPlaylist.PlaylistID == plist.ID,
                    tables.DjmdSongPlaylist.TrackNo >= track_no,
                )
                .order_by(tables.DjmdSongPlaylist.TrackNo)
            )
            for other_song in query:
                other_song.TrackNo += 1
                other_song.updated_at = now
                moved.append(other_song)
            self.registry.enable_tracking()

        # Add song to playlist
        song: tables.DjmdSongPlaylist = tables.DjmdSongPlaylist.create(
            ID=id_,
            PlaylistID=str(pid),
            ContentID=str(cid),
            TrackNo=track_no,
            UUID=uuid,
            created_at=now,
            updated_at=now,
        )
        self.add(song)
        if not insert_at_end:
            moved.append(song)
            self.registry.on_move(moved)

        return song
```