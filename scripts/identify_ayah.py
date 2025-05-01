#!/usr/bin/env python3
"""
Ayah identification script for the Quran Reciter Classifier system.
Given an audio file, identifies the Quranic verse (ayah).
"""
import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to Python path when running script directly
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.utils.ayah_identification import identify_ayah

# Initialize Rich console for pretty output
console = Console()

def parse_args():
    """Process command line options for ayah identification."""
    parser = argparse.ArgumentParser(
        description="Identify Quranic verse (ayah) from audio file")

    parser.add_argument(
        "audio", type=str,
        help="Path to audio file to analyze")

    parser.add_argument(
        "--model-type", type=str, default="whisper",
        help="Type of model to use (default: whisper)")

    parser.add_argument(
        "--show-debug", action="store_true",
        help="Show detailed debug information")

    parser.add_argument(
        "--min-confidence", type=float, default=0.70,
        help="Minimum confidence threshold (default: 0.70)")

    parser.add_argument(
        "--max-matches", type=int, default=5,
        help="Maximum number of similar matches to show (default: 5)")

    return parser.parse_args()

def format_ayah_display(ayah_data, confidence=None, transcription=None):
    """Format ayah data for display."""
    table = Table(show_header=False, box=None)
    table.add_column("Label", style="bold cyan")
    table.add_column("Value")

    table.add_row("Surah Number", str(ayah_data['surah_number']))
    table.add_row("Surah Name (Arabic)", ayah_data['surah_name'])
    table.add_row("Surah Name (English)", ayah_data['surah_name_en'])
    table.add_row("Ayah", str(ayah_data['ayah_number']))
    table.add_row("Text", ayah_data['ayah_text'])
    if transcription:
        table.add_row("Transcribed", transcription)
    if confidence is not None:
        table.add_row("Confidence", f"{confidence:.1%}")

    return table

def main():
    """Run ayah identification on an audio file."""
    args = parse_args()

    # Make sure the audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists() or not audio_path.is_file():
        console.print(f"[red]Error:[/red] Audio file not found: {args.audio}")
        return 1

    try:
        # Process audio file using shared logic
        console.print("\n[bold]Processing audio file...[/bold]")
        result = identify_ayah(
            audio_path,
            model_type=args.model_type,
            min_confidence=args.min_confidence,
            max_matches=args.max_matches
        )

        if result['error']:
            console.print(f"[red]Error:[/red] {result['error']}")
            return 1

        if args.show_debug:
            console.print(f"\n[dim]Transcription:[/dim] {result['transcription']}")
            console.print(f"[dim]Processing time:[/dim] {result['processing_time']:.2f}s")

        # Display results
        console.print("\n[bold green]Ayah Identification Results[/bold green]")
        
        if result['best_match']:
            best_match = result['best_match']
            panel = Panel(
                format_ayah_display(
                    best_match, 
                    best_match.get('confidence_score'),
                    result['transcription']
                ),
                title="[bold]Matched Ayah[/bold]",
                border_style="green"
            )
            console.print(panel)

            # Show similar matches if available
            if len(result['matches']) > 1:
                similar_table = Table(title="Similar Matches", show_lines=True)
                similar_table.add_column("Confidence", style="cyan")
                similar_table.add_column("Surah (Arabic)", style="bold")
                similar_table.add_column("Surah (English)")
                similar_table.add_column("Ayah")
                similar_table.add_column("Text")

                for match in result['matches'][1:]:  # Skip the best match
                    similar_table.add_row(
                        f"{match['confidence_score']:.1%}",
                        f"{match['surah_name']} ({match['surah_number']})",
                        match['surah_name_en'],
                        str(match['ayah_number']),
                        match['ayah_text']
                    )
                console.print(similar_table)
        else:
            console.print("[yellow]No reliable match found.[/yellow]")
            if args.show_debug and result['matches']:
                console.print("\n[dim]Top potential matches (low confidence):[/dim]")
                for match in result['matches']:
                    console.print(f"[dim]• {match['confidence_score']:.1%} - {match['surah_name']} ({match['surah_number']}:{match['ayah_number']}) - {match['surah_name_en']}[/dim]")

        return 0

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if args.show_debug:
            console.print_exception()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 