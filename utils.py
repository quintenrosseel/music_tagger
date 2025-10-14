"""Utility functions for RekordBox database operations."""

import datetime
from typing import List
from uuid import uuid4

import librosa
import numpy as np
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
        "SampleRate": pl.Int32,
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
                "SampleRate": int(content.SampleRate),
            }
            for content in db.get_content()
        ],
        schema=schema,
    )


def get_clean_songs(db: Rekordbox6Database, rename: bool = False) -> pl.DataFrame:
    """
    Get all songs, with:
    has_tags column
    Rekordbox data filtered out
    """
    out = (
        get_db_content(db)
        .with_columns(
            (
                (pl.col("MyTagNames").list.len() > 0)
                & ~pl.col("MyTagNames").list.contains("AUTOTAG")
            ).alias("has_tags")
        )
        .filter((pl.col("BPM") > 0) & (pl.col("ArtistName") != "rekordbox"))
    )
    if rename:
        return out.rename(
            mapping={
                "ContentID": "song_id",
                "FolderPath": "song_path",
                "Title": "song_title",
                "ArtistID": "artist_id",
                "ArtistName": "artist_name",
                "GenreID": "genre_id",
                "GenreName": "genre_name",
                "BPM": "bpm",
                "DateCreated": "date_created",
                "Length": "length",
                "MyTagIDs": "tag_ids",
                "MyTagNames": "tag_names",
                "SampleRate": "sample_rate",
            }
        )
    return out


def get_clean_tags(db: Rekordbox6Database) -> pl.DataFrame:
    """
    Return polars dataframe with

    - TagID
    - UUID
    - TagGroup
    - TagName
    """

    tag_table = tables.DjmdMyTag
    tags: List[tables.DjmdMyTag] = db.query(tag_table).all()

    tag_schema = {
        "TagID": pl.Utf8,
        "Seq": pl.Int32,
        "Attribute": pl.Int32,
        "Name": pl.Utf8,
        "ParentID": pl.Utf8,
        "UUID": pl.Utf8,
    }

    raw_tags_df = pl.from_records(
        [
            {
                "TagID": t.ID,
                "Seq": t.Seq,
                "Attribute": t.Attribute,
                "Name": t.Name,
                "ParentID": t.ParentID,
                "UUID": t.UUID,
            }
            for t in tags
        ],
        schema=tag_schema,
    )

    # Root groups are parent tags (e.g. Genre, Mood, Situation, Years & Origin)
    root_groups_df = raw_tags_df.filter(pl.col("ParentID") == "root")

    # Tags are the subgroups (e.g. Techno, House, Surfing, Sunset, etc. )
    filtered_tags_df = raw_tags_df.filter(pl.col("ParentID") != "root")

    return filtered_tags_df.join(
        root_groups_df.select(["TagID", "Name"]),
        left_on="ParentID",
        right_on="TagID",
        how="right",
    ).select(
        [
            pl.col("TagID"),
            pl.col("UUID"),
            pl.col("Name_right").alias("TagGroup"),
            pl.col("Name").alias("TagName"),
        ]
    )


def get_clean_eda(db: Rekordbox6Database) -> pl.DataFrame:
    """
    Get dataset for EDA & inference
    """
    clean_songs_df = get_clean_songs(db)
    clean_tags_df = get_clean_tags(db=db)

    return (
        clean_songs_df.explode(columns=["MyTagNames", "MyTagIDs"])  # one tag per song
        .drop("MyTagNames")
        .join(other=clean_tags_df, how="left", left_on="MyTagIDs", right_on="TagID")
        .rename(
            mapping={
                "ContentID": "song_id",
                "FolderPath": "song_path",
                "Title": "song_title",
                "ArtistID": "artist_id",
                "ArtistName": "artist_name",
                "GenreID": "genre_id",
                "GenreName": "genre_name",
                "BPM": "bpm",
                "DateCreated": "date_created",
                "Length": "length",
                "MyTagIDs": "tag_id",
                "UUID": "tag_uuid",
                "TagGroup": "tag_group",
                "TagName": "tag_name",
                "SampleRate": "sample_rate",
            }
        )
    )


def extract_audio_features(audio, sr, fast_mode=False):
    """Extract comprehensive audio features for ML models.

    Parameters
    ----------
    audio : np.ndarray
        Audio time series loaded with librosa
    sr : int
        Sample rate of the audio
    fast_mode : bool, optional
        If True, skip expensive computations (HPSS, tonnetz).
        Reduces feature count but ~3-5x faster. Default: False

    Returns
    -------
    features : dict
        Dictionary containing extracted features. Keys are feature names,
        values are either scalars or numpy arrays.
        Total feature vector length: ~380 dimensions (full), ~360 (fast)
    """
    features = {}

    # Compute STFT once and reuse (major optimization)
    stft = librosa.stft(audio)
    S = np.abs(stft)

    # 1. Mel Spectrogram (128 bands) - compute from power spectrogram
    mel_spec = librosa.feature.melspectrogram(S=S**2, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features["mel_spec_mean"] = np.mean(mel_spec_db, axis=1)  # Shape: (128,)
    features["mel_spec_std"] = np.std(mel_spec_db, axis=1)  # Shape: (128,)

    # 2. MFCCs - compute from mel spectrogram
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), sr=sr, n_mfcc=20)
    features["mfcc_mean"] = np.mean(mfccs, axis=1)  # Shape: (20,)
    features["mfcc_std"] = np.std(mfccs, axis=1)  # Shape: (20,)

    # 3. Chroma features - from STFT
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    features["chroma_mean"] = np.mean(chroma, axis=1)  # Shape: (12,)
    features["chroma_std"] = np.std(chroma, axis=1)  # Shape: (12,)

    # 4. Spectral features - all from STFT
    spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_centroid_std"] = np.std(spectral_centroid)

    spectral_rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
    features["spectral_rolloff_std"] = np.std(spectral_rolloff)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
    features["spectral_bandwidth_std"] = np.std(spectral_bandwidth)

    spectral_contrast = librosa.feature.spectral_contrast(S=S, sr=sr, n_bands=6)
    features["spectral_contrast_mean"] = np.mean(
        spectral_contrast, axis=1
    )  # Shape: (7,)
    features["spectral_contrast_std"] = np.std(spectral_contrast, axis=1)  # Shape: (7,)

    spectral_flatness = librosa.feature.spectral_flatness(S=S)
    features["spectral_flatness_mean"] = np.mean(spectral_flatness)
    features["spectral_flatness_std"] = np.std(spectral_flatness)

    # 5. Tempo and beat features
    onset_env = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(S, ref=np.max), sr=sr
    )
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    features["tempo"] = tempo
    features["beat_count"] = len(beats)

    # Beat strength/regularity
    if len(beats) > 1:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        beat_intervals = np.diff(beat_times)
        features["beat_regularity"] = 1.0 / (np.std(beat_intervals) + 1e-6)
    else:
        features["beat_regularity"] = 0.0

    # 6. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    # 7. Energy and dynamics features (important for mood/situation)
    rms = librosa.feature.rms(S=S)
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    features["rms_max"] = np.max(rms)
    features["dynamic_range"] = np.max(rms) - np.min(rms)

    # Energy variation over time (for detecting build-ups, drops)
    rms_diff = np.diff(rms[0])
    features["rms_delta_mean"] = np.mean(np.abs(rms_diff))
    features["rms_delta_std"] = np.std(rms_diff)

    # 8. Onset features (rhythm/energy) - reuse onset_env from tempo
    features["onset_strength_mean"] = np.mean(onset_env)
    features["onset_strength_std"] = np.std(onset_env)

    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration = librosa.get_duration(y=audio, sr=sr)
    features["onset_rate"] = len(onset_frames) / duration if duration > 0 else 0

    if not fast_mode:
        # 9. Harmonic and percussive separation (EXPENSIVE - skip in fast mode)
        y_harmonic, y_percussive = librosa.effects.hpss(audio, margin=2.0)

        harmonic_energy = np.sum(y_harmonic**2)
        percussive_energy = np.sum(y_percussive**2)
        features["harmonic_percussive_ratio"] = harmonic_energy / (
            percussive_energy + 1e-6
        )
        features["percussive_strength"] = percussive_energy / (
            harmonic_energy + percussive_energy + 1e-6
        )

        # 10. Tonnetz features (requires harmonic - skip in fast mode)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        features["tonnetz_mean"] = np.mean(tonnetz, axis=1)  # Shape: (6,)
        features["tonnetz_std"] = np.std(tonnetz, axis=1)  # Shape: (6,)
    else:
        # Placeholder values for fast mode
        features["harmonic_percussive_ratio"] = 0.0
        features["percussive_strength"] = 0.0
        features["tonnetz_mean"] = np.zeros(6)
        features["tonnetz_std"] = np.zeros(6)

    # 11. Temporal features - analyze first/middle/last sections
    segment_size = len(audio) // 3
    sections = [
        audio[:segment_size],
        audio[segment_size : 2 * segment_size],
        audio[2 * segment_size :],
    ]

    section_energies = [np.mean(librosa.feature.rms(y=section)) for section in sections]
    features["energy_start"] = section_energies[0]
    features["energy_middle"] = section_energies[1]
    features["energy_end"] = section_energies[2]
    features["energy_increase_ratio"] = section_energies[2] / (
        section_energies[0] + 1e-6
    )

    return features


def features_dict_to_flat_dict(features):
    """Flatten nested feature dictionary to a single-level dictionary.

    Converts arrays to individual columns (e.g., mel_spec_mean[0], mel_spec_mean[1], ...).

    Parameters
    ----------
    features : dict
        Feature dictionary from extract_audio_features()

    Returns
    -------
    flat_dict : dict
        Flattened dictionary with all features as scalars
    """
    flat = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            # Flatten arrays into individual columns
            for i, v in enumerate(value):
                flat[f"{key}_{i}"] = float(v)
        else:
            flat[key] = (
                float(value) if isinstance(value, (np.number, np.ndarray)) else value
            )
    return flat


def get_aggregate_tags_dataset(db: Rekordbox6Database, top_k: int = 20) -> pl.DataFrame:
    """
    Get clean EDA dataset with top 20 most occurring tag sets per group, per song.

    This dataset is designed to help create "aggregate_tags" - single labels that can
    be extrapolated per song consisting of one or more tags from that group.

    Years & Origin tags are filtered out since they cannot be learned from audio features.

    Parameters
    ----------
    db : Rekordbox6Database
        The RekordBox database instance to use for the operation.
    top_k : int, optional
        Number of top tag sets to return per group (default: 20)

    Returns
    -------
    aggregate_tags_df : pl.DataFrame
        DataFrame with columns:
        - tag_group: The tag group (Genre, Mood, Situation)
        - tag_set: Comma-separated string of tags that appear together
        - tag_ids_list: List[str] of tag IDs in the set (sorted)
        - tag_names_list: List[str] of tag names in the set (sorted)
        - song_ids_list: List[str] of song IDs that have this tag combination
        - song_count: Number of songs with this tag combination
        - rank: Rank within the tag group (1 to top_k)

        Only includes the top_k most common tag sets per group.

    Examples
    --------
    >>> db = Rekordbox6Database()
    >>> agg_tags = get_aggregate_tags_dataset(db)
    >>> agg_tags.filter(pl.col("tag_group") == "Genre").head()

    # To enrich songs with their aggregate labels:
    >>> enriched = agg_tags.explode("song_ids_list").rename({"song_ids_list": "song_id"})
    >>> songs_with_labels = songs_df.join(enriched, on="song_id", how="left")
    """
    clean_eda_df = get_clean_eda(db)

    # Filter out Years & Origin tags
    filtered_eda = clean_eda_df.filter(
        (pl.col("tag_group").is_not_null()) & (pl.col("tag_group") != "Years & Origin")
    )

    # For each song and tag group, collect the set of tags
    tag_sets_per_song = (
        filtered_eda.group_by("song_id", "tag_group")
        .agg(
            pl.col("tag_name").sort().alias("tag_names_list"),
            pl.col("tag_id").sort().alias("tag_ids_list"),
        )
        .with_columns(pl.col("tag_names_list").list.join(", ").alias("tag_set"))
    )

    # Count how often each tag set appears per group and collect song IDs
    tag_set_counts = (
        tag_sets_per_song.group_by(
            "tag_group", "tag_set", "tag_names_list", "tag_ids_list"
        )
        .agg(
            pl.col("song_id").alias("song_ids_list"),
            pl.len().alias("song_count"),
        )
        .sort(["tag_group", "song_count"], descending=[False, True])
    )

    # Add rank per group and filter top_k
    result = (
        tag_set_counts.with_columns(
            pl.col("song_count")
            .rank(method="ordinal", descending=True)
            .over("tag_group")
            .alias("rank")
        )
        .filter(pl.col("rank") <= top_k)
        .sort(["tag_group", "rank"])
        .select(
            [
                "tag_group",
                "tag_set",
                "tag_ids_list",
                "tag_names_list",
                "song_ids_list",
                "song_count",
                "rank",
            ]
        )
    )

    return result


def extract_features_for_songs(
    songs_df, fast_mode=True, sr=44100, checkpoint_path=None, checkpoint_every=100
):
    """Extract audio features for all songs in a DataFrame with checkpointing.

    Parameters
    ----------
    songs_df : pl.DataFrame
        DataFrame with at minimum 'song_id' and 'song_path' columns
    fast_mode : bool, optional
        Use fast feature extraction (default: True)
    sr : int, optional
        Sample rate for loading audio (default: 44100)
    checkpoint_path : str, optional
        Path to save checkpoints (parquet file). If provided, will save progress
        every checkpoint_every songs and resume from last checkpoint if it exists.
    checkpoint_every : int, optional
        Save checkpoint every N songs (default: 100)

    Returns
    -------
    features_df : pl.DataFrame
        DataFrame with song_id, song_path, and all extracted features as columns
    """
    from pathlib import Path

    try:
        from tqdm import tqdm

        has_tqdm = True
    except ImportError:
        has_tqdm = False
        print("Install tqdm for progress bars: pip install tqdm")

    results = []
    processed_ids = set()
    failed_files = []

    # Load existing checkpoint if it exists
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_df = pl.read_parquet(checkpoint_path)
        results = checkpoint_df.to_dicts()
        processed_ids = set(checkpoint_df["song_id"].to_list())
        print(f"Resuming from checkpoint: {len(processed_ids)} songs already processed")

    # Filter out already processed songs
    remaining_df = songs_df.filter(~pl.col("song_id").is_in(processed_ids))
    print(f"Processing {len(remaining_df)} songs (total: {len(songs_df)})")

    # Setup progress bar
    iterator = remaining_df.iter_rows(named=True)
    if has_tqdm:
        iterator = tqdm(iterator, total=len(remaining_df), desc="Extracting features")

    for idx, row in enumerate(iterator):
        song_id = row["song_id"]
        song_path = row["song_path"]

        try:
            # Load audio
            audio, _ = librosa.load(song_path, sr=sr)

            # Extract features
            features = extract_audio_features(audio, sr, fast_mode=fast_mode)

            # Flatten to single-level dict
            flat_features = features_dict_to_flat_dict(features)

            # Add song metadata
            flat_features["song_id"] = song_id
            flat_features["song_path"] = song_path

            results.append(flat_features)

            # Save checkpoint periodically
            if checkpoint_path and (idx + 1) % checkpoint_every == 0:
                try:
                    # Ensure directory exists
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    temp_df = pl.DataFrame(results)
                    temp_df.write_parquet(checkpoint_path)
                    if has_tqdm:
                        tqdm.write(f"Checkpoint saved: {len(results)} songs processed")
                    else:
                        print(f"Checkpoint saved: {len(results)} songs processed")
                except Exception as e:
                    if has_tqdm:
                        tqdm.write(f"⚠️  Checkpoint save failed: {e}")
                    else:
                        print(f"⚠️  Checkpoint save failed: {e}")

        except PermissionError:
            failed_files.append(
                {
                    "song_id": song_id,
                    "song_path": song_path,
                    "error": "Permission denied",
                }
            )
            continue
        except Exception as e:
            failed_files.append(
                {"song_id": song_id, "song_path": song_path, "error": type(e).__name__}
            )
            continue

    # Final save if using checkpoints
    if checkpoint_path and results:
        try:
            # Ensure directory exists
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            final_df = pl.DataFrame(results)
            final_df.write_parquet(checkpoint_path)
            print(f"Final checkpoint saved: {len(results)} songs")
        except Exception as e:
            print(f"⚠️  Failed to save final checkpoint: {e}")
            print("⚠️  Data is still in memory and will be returned")

    # Log failed files
    if failed_files:
        print(f"\n{len(failed_files)} files failed to process:")
        for failed in failed_files:
            print(f"  - {Path(failed['song_path']).name}: {failed['error']}")

    # Convert to Polars DataFrame
    if not results:
        raise ValueError("No features extracted successfully")

    df = pl.DataFrame(results)
    # Reorder columns: song_id, song_path, then features
    feature_cols = [c for c in df.columns if c not in ["song_id", "song_path"]]
    return df.select(["song_id", "song_path"] + feature_cols)


def get_base_dataset(
    db: Rekordbox6Database, min_tag_count: int | None = None
) -> pl.DataFrame:
    """
    Returns the base dataset for multi-label classification.

    Each row represents a song-tag pair, with tags filtered to exclude
    "Years & Origin" category since those are difficult to predict from audio.
    Tags can also be filtered by minimum occurrence count.

    Parameters
    ----------
    db : Rekordbox6Database
        The RekordBox database instance to use for the operation.
    min_tag_count : int | None, optional
        Minimum number of occurrences required for a tag to be included.
        Tags with fewer occurrences will be filtered out. If None (default),
        all tags are included without filtering by count.

    Returns
    -------
    base_df : pl.DataFrame
        DataFrame with columns:
        - song_id: Song identifier
        - song_path: Path to the audio file
        - song_title: Title of the song
        - artist_id: Artist identifier
        - artist_name: Artist name
        - genre_id: Genre identifier
        - genre_name: Genre name
        - bpm: Beats per minute
        - date_created: Date song was added
        - length: Song length in seconds
        - tag_ids: List of all tag IDs for the song
        - tag_names: List of all tag names for the song
        - sample_rate: Audio sample rate
        - has_tags: Boolean indicating if song has tags
        - tag_id: Individual tag ID (one per row)
        - tag_name: Individual tag name (one per row)
        - tag_group: Tag category (Genre, Mood, or Situation)

        Songs without tags or with only "Years & Origin" tags will have
        null values for tag_id, tag_name, and tag_group.

    Examples
    --------
    >>> db = Rekordbox6Database()
    >>> # Get all tags without filtering
    >>> base = get_base_dataset(db)
    >>> # Get base dataset with tags that appear at least 10 times
    >>> base = get_base_dataset(db, min_tag_count=10)
    >>> # Get all unique tags per group
    >>> base.filter(pl.col("tag_group").is_not_null()).group_by("tag_group", "tag_name").count()
    """
    clean_songs_df = get_clean_songs(db, rename=True)
    clean_tags_df = get_clean_tags(db)

    # Filter tags to exclude Years & Origin
    filtered_tags_df = clean_tags_df.filter(pl.col("TagGroup") != "Years & Origin")

    # Join songs with their individual tags (exploded)
    base_df = (
        clean_songs_df.explode(columns=["tag_names", "tag_ids"])
        .join(
            filtered_tags_df.select(["TagID", "TagName", "TagGroup"]),
            left_on="tag_ids",
            right_on="TagID",
            how="left",
        )
        .rename(
            {
                "tag_ids": "tag_id",
                "tag_names": "tag_name_original",
                "TagName": "tag_name",
                "TagGroup": "tag_group",
            }
        )
        .filter(  # Filter out Years & Origin tags
            (pl.col("tag_group").is_not_null())
            & (pl.col("tag_name").is_not_null())
            & (pl.col("tag_id").is_not_null())
        )
        .drop("tag_name_original")
    )

    # Filter by minimum tag count if specified
    if min_tag_count is not None:
        # Count occurrences per tag
        tag_counts = (
            base_df.group_by(["tag_name", "tag_group"])
            .agg(pl.len().alias("count"))
            .sort("count")
        )

        # Identify tags below threshold
        tags_to_remove = tag_counts.filter(pl.col("count") < min_tag_count)

        if len(tags_to_remove) > 0:
            print(f"\nFiltering tags with fewer than {min_tag_count} occurrences:")
            print(f"{'='*60}")

            # Group by tag_group for cleaner output
            for group in tags_to_remove["tag_group"].unique().sort():
                group_tags = tags_to_remove.filter(pl.col("tag_group") == group)
                print(f"\n{group}:")
                for row in group_tags.iter_rows(named=True):
                    print(f"  - {row['tag_name']}: {row['count']} occurrence(s)")

            print(f"\n{'='*60}")
            print(f"Total tags filtered: {len(tags_to_remove)}")
            print(f"{'='*60}\n")

            # Keep only tags that meet the threshold
            valid_tags = (
                tag_counts.filter(pl.col("count") >= min_tag_count).select(
                    ["tag_name", "tag_group"]
                )
            )

            # Filter base_df to only include valid tags
            base_df = base_df.join(valid_tags, on=["tag_name", "tag_group"], how="inner")

    return base_df.sort("song_id")
