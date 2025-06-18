using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Encodings.Web;

namespace MajdataEdit
{
    class Program
    {
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.Error.WriteLine("Usage: SimaiSerializerFromMajdataEdit.exe <file_or_directory_path>");
                Console.Error.WriteLine("Processes a single maidata.txt file or all maidata.txt files in a directory and outputs JSON to stdout.");
                Environment.Exit(1);
                return;
            }

            string inputPath = args[0];
            var filesToProcess = new List<string>();

            if (File.Exists(inputPath))
            {
                if (Path.GetExtension(inputPath).Equals(".txt", StringComparison.OrdinalIgnoreCase))
                {
                    filesToProcess.Add(inputPath);
                }
            }
            else if (Directory.Exists(inputPath))
            {
                foreach (string filePath in Directory.GetFiles(inputPath, "maidata.txt", SearchOption.AllDirectories))
                {
                    filesToProcess.Add(filePath);
                }
            }
            else
            {
                Console.Error.WriteLine($"Error: Input path not found: {inputPath}");
                Environment.Exit(1);
                return;
            }

            if (filesToProcess.Count == 0)
            {
                Console.Error.WriteLine($"No maidata.txt files found to process in path: {inputPath}");
                return;
            }

            var allCharts = new List<object>();
            foreach (var file in filesToProcess)
            {
                var chartsFromFile = ProcessFile(file);
                if (chartsFromFile != null)
                {
                    allCharts.AddRange(chartsFromFile);
                }
            }

            var options = new JsonSerializerOptions 
            { 
                WriteIndented = true, 
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping 
            };
            string jsonString = JsonSerializer.Serialize(allCharts, options);
            Console.WriteLine(jsonString);
        }

        static List<object> ProcessFile(string filePath)
        {
            ClearFumens();
            if (!SimaiProcess.ReadData(filePath))
            {
                Console.Error.WriteLine($"Warning: Failed to read or process file, skipping: {filePath}");
                return null;
            }

            string songId = Path.GetFileName(Path.GetDirectoryName(filePath));
            var chartsForFile = new List<object>();

            for (int level_index = 0; level_index < SimaiProcess.fumens.Length; level_index++)
            {
                if (!string.IsNullOrEmpty(SimaiProcess.fumens[level_index]))
                {
                    string rawFumenText = SimaiProcess.fumens[level_index];
                    SimaiProcess.Serialize(rawFumenText);

                    foreach (var note in SimaiProcess.notelist)
                    {
                        note.noteList = note.getNotes();
                    }

                    var noteDataForLevel = new List<object>();
                    foreach (var noteGroup in SimaiProcess.notelist)
                    {
                        var noteGroupData = new
                        {
                            Time = noteGroup.time,
                            Notes = new List<Dictionary<string, object>>()
                        };

                        foreach (var note in noteGroup.noteList)
                        {
                            var noteProperties = new Dictionary<string, object>
                            {
                                { "holdTime", note.holdTime },
                                { "isBreak", note.isBreak },
                                { "isEx", note.isEx },
                                { "isFakeRotate", note.isFakeRotate },
                                { "isForceStar", note.isForceStar },
                                { "isHanabi", note.isHanabi },
                                { "isSlideBreak", note.isSlideBreak },
                                { "isSlideNoHead", note.isSlideNoHead },
                                { "noteContent", note.noteContent ?? string.Empty },
                                { "noteType", note.noteType.ToString() },
                                { "slideStartTime", note.slideStartTime },
                                { "slideTime", note.slideTime },
                                { "startPosition", note.startPosition },
                                { "touchArea", note.touchArea }
                            };
                            noteGroupData.Notes.Add(noteProperties);
                        }
                        noteDataForLevel.Add(noteGroupData);
                    }

                    var chartOutput = new
                    {
                        song_id = songId,
                        level_index = level_index,
                        notes = noteDataForLevel
                    };
                    chartsForFile.Add(chartOutput);
                }
            }
            return chartsForFile;
        }

        static void ClearFumens()
        {
            for (int i = 0; i < SimaiProcess.fumens.Length; i++)
            {
                SimaiProcess.fumens[i] = "";
            }
        }
    }
}
