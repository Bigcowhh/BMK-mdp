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
            if (args.Length < 2)
            {
                Console.Error.WriteLine("Usage: SimaiSerializerFromMajdataEdit.exe <file_or_directory_path> <output_directory>");
                Console.Error.WriteLine("Processes a single maidata.txt file or all maidata.txt files in a directory and outputs individual JSON files for each difficulty level.");
                Environment.Exit(1);
                return;
            }
            string inputPath = args[0];
            string outputPath = args[1];

            if (!Directory.Exists(outputPath))
            {
                Directory.CreateDirectory(outputPath);
            }

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

            foreach (var file in filesToProcess)
            {
                ProcessFile(file, outputPath);
            }
        }
        static void ProcessFile(string filePath, string outputPath)
        {
            ClearFumens();
            if (!SimaiProcess.ReadData(filePath))
            {
                Console.Error.WriteLine($"Warning: Failed to read or process file, skipping: {filePath}");
                return;
            }

            string folderName = Path.GetFileName(Path.GetDirectoryName(filePath));
            string songId = ExtractNumericId(folderName);

            if (songId == null)
            {
                Console.Error.WriteLine($"Error: Cannot extract numeric ID from folder name '{folderName}', skipping: {filePath}");
                return;
            }

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping
            };

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

                    string fileName = $"{folderName}_{level_index}.json";
                    string outputFilePath = Path.Combine(outputPath, fileName);
                    string jsonString = JsonSerializer.Serialize(chartOutput, options);
                    File.WriteAllText(outputFilePath, jsonString);
                    Console.WriteLine($"Generated: {outputFilePath}");
                }
            }
        }

        static string ExtractNumericId(string folderName)
        {
            if (string.IsNullOrEmpty(folderName))
            {
                return null;
            }

            int underscoreIndex = folderName.IndexOf('_');
            if (underscoreIndex > 0)
            {
                string numericPart = folderName.Substring(0, underscoreIndex);
                if (int.TryParse(numericPart, out _))
                {
                    return numericPart;
                }
            }
            return null;
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
