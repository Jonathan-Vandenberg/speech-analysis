-- CreateSchema
CREATE SCHEMA IF NOT EXISTS "public";

-- CreateEnum
CREATE TYPE "public"."UserRole" AS ENUM ('TEACHER', 'ADMIN', 'STUDENT', 'PARENT');

-- CreateEnum
CREATE TYPE "public"."LanguageType" AS ENUM ('ENGLISH', 'VIETNAMESE', 'JAPANESE', 'SPANISH', 'ITALIAN', 'FRENCH', 'GERMAN', 'PORTUGESE');

-- CreateEnum
CREATE TYPE "public"."AssignmentType" AS ENUM ('CLASS', 'INDIVIDUAL');

-- CreateEnum
CREATE TYPE "public"."LanguageAssessmentType" AS ENUM ('SCRIPTED_US', 'SCRIPTED_UK', 'UNSCRIPTED_US', 'UNSCRIPTED_UK', 'PRONUNCIATION_US', 'PRONUNCIATION_UK');

-- CreateEnum
CREATE TYPE "public"."EvaluationType" AS ENUM ('CUSTOM', 'IMAGE', 'VIDEO', 'Q_AND_A', 'READING', 'PRONUNCIATION');

-- CreateEnum
CREATE TYPE "public"."ActivityLogType" AS ENUM ('USER_CREATED', 'USER_UPDATED', 'USER_DELETED', 'USER_BLOCKED', 'USER_UNBLOCKED', 'USER_CONFIRMED', 'USER_PASSWORD_CHANGED', 'USER_ROLE_CHANGED', 'CLASS_CREATED', 'CLASS_UPDATED', 'CLASS_DELETED', 'CLASS_USERS_ADDED', 'CLASS_USERS_REMOVED', 'ASSIGNMENT_CREATED', 'ASSIGNMENT_UPDATED', 'ASSIGNMENT_DELETED', 'ASSIGNMENT_PUBLISHED', 'ASSIGNMENT_ARCHIVED', 'INDIVIDUAL_ASSIGNMENT_CREATED', 'INDIVIDUAL_ASSIGNMENT_DELETED', 'USER_LOGIN', 'USER_LOGOUT', 'USER_LOGIN_FAILED', 'SYSTEM_BACKUP_CREATED', 'SYSTEM_MAINTENANCE', 'STUDENT_CREATED', 'TEACHER_CREATED');

-- CreateEnum
CREATE TYPE "public"."AssignmentCategoryType" AS ENUM ('IMAGE', 'VIDEO', 'Q_AND_A', 'CUSTOM', 'READING', 'PRONUNCIATION', 'Q_AND_A_IMAGE');

-- CreateEnum
CREATE TYPE "public"."ToolType" AS ENUM ('PLANNING', 'ASSESSMENT', 'RESOURCES', 'ADMIN', 'PUPIL_REPORTS', 'LEADERSHIP', 'WELLBEING');

-- CreateEnum
CREATE TYPE "public"."DashboardSnapshotType" AS ENUM ('daily', 'weekly', 'monthly');

-- CreateEnum
CREATE TYPE "public"."PerformanceMetricType" AS ENUM ('COMPLETION_RATE', 'ACCURACY_RATE', 'AVERAGE_SCORE', 'ACTIVE_USERS', 'QUESTIONS_ANSWERED', 'TIME_SPENT', 'LOGIN_COUNT', 'ASSIGNMENT_SUBMISSIONS');

-- CreateEnum
CREATE TYPE "public"."EntityType" AS ENUM ('SCHOOL', 'CLASS', 'TEACHER', 'STUDENT', 'ASSIGNMENT', 'QUESTION');

-- CreateEnum
CREATE TYPE "public"."TimeFrame" AS ENUM ('HOURLY', 'DAILY', 'WEEKLY', 'MONTHLY');

-- CreateTable
CREATE TABLE "public"."users" (
    "id" TEXT NOT NULL,
    "username" TEXT NOT NULL,
    "email" TEXT NOT NULL,
    "provider" TEXT,
    "password" TEXT,
    "resetPasswordToken" TEXT,
    "confirmationToken" TEXT,
    "confirmed" BOOLEAN NOT NULL DEFAULT false,
    "blocked" BOOLEAN NOT NULL DEFAULT false,
    "customRole" "public"."UserRole" NOT NULL,
    "address" TEXT,
    "customImage" TEXT,
    "phone" TEXT,
    "isPlayGame" BOOLEAN DEFAULT false,
    "theme" TEXT DEFAULT 'system',
    "averageScoreOfCompleted" DOUBLE PRECISION,
    "totalAssignments" INTEGER DEFAULT 0,
    "totalAssignmentsCompleted" INTEGER DEFAULT 0,
    "averageCompletionPercentage" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "roleId" TEXT,

    CONSTRAINT "users_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."roles" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "type" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "roles_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."permissions" (
    "id" TEXT NOT NULL,
    "action" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "permissions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."role_permissions" (
    "roleId" TEXT NOT NULL,
    "permissionId" TEXT NOT NULL,

    CONSTRAINT "role_permissions_pkey" PRIMARY KEY ("roleId","permissionId")
);

-- CreateTable
CREATE TABLE "public"."classes" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),

    CONSTRAINT "classes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."user_classes" (
    "userId" TEXT NOT NULL,
    "classId" TEXT NOT NULL,

    CONSTRAINT "user_classes_pkey" PRIMARY KEY ("userId","classId")
);

-- CreateTable
CREATE TABLE "public"."languages" (
    "id" TEXT NOT NULL,
    "language" "public"."LanguageType" NOT NULL,
    "code" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),

    CONSTRAINT "languages_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."assignments" (
    "id" TEXT NOT NULL,
    "topic" TEXT,
    "color" TEXT,
    "vocabularyItems" JSONB,
    "scheduledPublishAt" TIMESTAMP(3),
    "isActive" BOOLEAN DEFAULT true,
    "type" "public"."AssignmentType",
    "videoUrl" TEXT,
    "videoTranscript" TEXT,
    "languageAssessmentType" "public"."LanguageAssessmentType",
    "isIELTS" BOOLEAN DEFAULT false,
    "context" TEXT,
    "totalStudentsInScope" INTEGER DEFAULT 0,
    "completedStudentsCount" INTEGER DEFAULT 0,
    "completionRate" DOUBLE PRECISION,
    "averageScoreOfCompleted" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),
    "teacherId" TEXT,
    "languageId" TEXT,
    "dueDate" TIMESTAMP(3),

    CONSTRAINT "assignments_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."class_assignments" (
    "classId" TEXT NOT NULL,
    "assignmentId" TEXT NOT NULL,

    CONSTRAINT "class_assignments_pkey" PRIMARY KEY ("classId","assignmentId")
);

-- CreateTable
CREATE TABLE "public"."user_assignments" (
    "userId" TEXT NOT NULL,
    "assignmentId" TEXT NOT NULL,

    CONSTRAINT "user_assignments_pkey" PRIMARY KEY ("userId","assignmentId")
);

-- CreateTable
CREATE TABLE "public"."evaluation_settings" (
    "id" TEXT NOT NULL,
    "type" "public"."EvaluationType" NOT NULL,
    "customPrompt" TEXT,
    "rules" JSONB,
    "acceptableResponses" JSONB,
    "feedbackSettings" JSONB NOT NULL DEFAULT '{}',
    "assignmentId" TEXT NOT NULL,

    CONSTRAINT "evaluation_settings_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."questions" (
    "id" TEXT NOT NULL,
    "image" TEXT,
    "textQuestion" TEXT,
    "videoUrl" TEXT,
    "textAnswer" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),
    "assignmentId" TEXT NOT NULL,

    CONSTRAINT "questions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."quizzes" (
    "id" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "topic" TEXT NOT NULL,
    "description" TEXT,
    "numberOfQuestions" INTEGER NOT NULL,
    "numberOfOptions" INTEGER NOT NULL DEFAULT 4,
    "isAIGenerated" BOOLEAN NOT NULL DEFAULT false,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "timeLimitMinutes" INTEGER,
    "isLiveSession" BOOLEAN NOT NULL DEFAULT false,
    "liveSessionStartedAt" TIMESTAMP(3),
    "liveSessionEndedAt" TIMESTAMP(3),
    "scheduledPublishAt" TIMESTAMP(3),
    "dueDate" TIMESTAMP(3),
    "currentSession" INTEGER NOT NULL DEFAULT 1,
    "allowMultipleSessions" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),
    "teacherId" TEXT NOT NULL,

    CONSTRAINT "quizzes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."quiz_questions" (
    "id" TEXT NOT NULL,
    "question" TEXT NOT NULL,
    "correctAnswer" TEXT NOT NULL,
    "explanation" TEXT,
    "imageUrl" TEXT,
    "order" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "quizId" TEXT NOT NULL,

    CONSTRAINT "quiz_questions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."quiz_options" (
    "id" TEXT NOT NULL,
    "text" TEXT NOT NULL,
    "isCorrect" BOOLEAN NOT NULL DEFAULT false,
    "order" INTEGER NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "questionId" TEXT NOT NULL,

    CONSTRAINT "quiz_options_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."quiz_classes" (
    "quizId" TEXT NOT NULL,
    "classId" TEXT NOT NULL,

    CONSTRAINT "quiz_classes_pkey" PRIMARY KEY ("quizId","classId")
);

-- CreateTable
CREATE TABLE "public"."quiz_students" (
    "quizId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,

    CONSTRAINT "quiz_students_pkey" PRIMARY KEY ("quizId","userId")
);

-- CreateTable
CREATE TABLE "public"."quiz_submissions" (
    "id" TEXT NOT NULL,
    "sessionNumber" INTEGER NOT NULL DEFAULT 1,
    "score" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "totalScore" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "percentage" DOUBLE PRECISION NOT NULL DEFAULT 0,
    "isCompleted" BOOLEAN NOT NULL DEFAULT false,
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" TIMESTAMP(3),
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "quizId" TEXT NOT NULL,
    "studentId" TEXT NOT NULL,

    CONSTRAINT "quiz_submissions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."quiz_answers" (
    "id" TEXT NOT NULL,
    "answer" TEXT NOT NULL,
    "isCorrect" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "submissionId" TEXT NOT NULL,
    "questionId" TEXT NOT NULL,

    CONSTRAINT "quiz_answers_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."quiz_live_sessions" (
    "id" TEXT NOT NULL,
    "startedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "endedAt" TIMESTAMP(3),
    "timeLimitMinutes" INTEGER,
    "isActive" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "quizId" TEXT NOT NULL,
    "teacherId" TEXT NOT NULL,

    CONSTRAINT "quiz_live_sessions_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."quiz_live_student_progress" (
    "id" TEXT NOT NULL,
    "currentQuestion" INTEGER NOT NULL DEFAULT 1,
    "questionsAnswered" INTEGER NOT NULL DEFAULT 0,
    "isCompleted" BOOLEAN NOT NULL DEFAULT false,
    "joinedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "lastActivity" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "sessionId" TEXT NOT NULL,
    "studentId" TEXT NOT NULL,

    CONSTRAINT "quiz_live_student_progress_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."student_assignment_progress" (
    "id" TEXT NOT NULL,
    "isComplete" BOOLEAN NOT NULL DEFAULT false,
    "isCorrect" BOOLEAN NOT NULL DEFAULT false,
    "languageConfidenceResponse" JSONB,
    "grammarCorrected" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),
    "studentId" TEXT NOT NULL,
    "assignmentId" TEXT NOT NULL,
    "questionId" TEXT,

    CONSTRAINT "student_assignment_progress_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."activity_logs" (
    "id" TEXT NOT NULL,
    "type" "public"."ActivityLogType" NOT NULL,
    "action" TEXT,
    "details" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),
    "userId" TEXT,
    "classId" TEXT,
    "assignmentId" TEXT,
    "quizId" TEXT,

    CONSTRAINT "activity_logs_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."assignment_categories" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "type" "public"."AssignmentCategoryType",
    "description" TEXT,
    "defaultPrompt" TEXT,
    "defaultRules" JSONB,
    "defaultFeedbackSettings" JSONB,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "isIELTS" BOOLEAN DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),
    "assignmentGroupId" TEXT,

    CONSTRAINT "assignment_categories_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."assignment_groups" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "color" TEXT NOT NULL,
    "isEnabled" BOOLEAN NOT NULL DEFAULT true,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),

    CONSTRAINT "assignment_groups_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."tools" (
    "id" TEXT NOT NULL,
    "type" "public"."ToolType" NOT NULL,
    "name" TEXT,
    "description" TEXT,
    "enabled" BOOLEAN NOT NULL DEFAULT true,
    "imageId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),

    CONSTRAINT "tools_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."sprite_sets" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "description" TEXT,
    "difficulty" INTEGER,
    "order" INTEGER NOT NULL,
    "stages" JSONB NOT NULL DEFAULT '[]',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),

    CONSTRAINT "sprite_sets_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."student_sprites" (
    "id" TEXT NOT NULL,
    "currentEvolutionStage" INTEGER NOT NULL DEFAULT 0,
    "completedAssignmentsCount" INTEGER NOT NULL DEFAULT 0,
    "currentSpriteSetIndex" INTEGER NOT NULL DEFAULT 0,
    "completedSpriteSets" JSONB NOT NULL DEFAULT '[]',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),
    "studentId" TEXT NOT NULL,
    "spriteSetId" TEXT,

    CONSTRAINT "student_sprites_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."stats_classes" (
    "id" TEXT NOT NULL,
    "averageCompletion" DOUBLE PRECISION,
    "averageScore" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),
    "classId" TEXT NOT NULL,

    CONSTRAINT "stats_classes_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."assignment_stats" (
    "id" TEXT NOT NULL,
    "assignmentId" TEXT NOT NULL,
    "totalStudents" INTEGER NOT NULL DEFAULT 0,
    "completedStudents" INTEGER NOT NULL DEFAULT 0,
    "inProgressStudents" INTEGER NOT NULL DEFAULT 0,
    "notStartedStudents" INTEGER NOT NULL DEFAULT 0,
    "completionRate" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "averageScore" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "totalQuestions" INTEGER NOT NULL DEFAULT 0,
    "totalAnswers" INTEGER NOT NULL DEFAULT 0,
    "totalCorrectAnswers" INTEGER NOT NULL DEFAULT 0,
    "accuracyRate" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "lastUpdated" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "assignment_stats_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."student_stats" (
    "id" TEXT NOT NULL,
    "studentId" TEXT NOT NULL,
    "totalAssignments" INTEGER NOT NULL DEFAULT 0,
    "completedAssignments" INTEGER NOT NULL DEFAULT 0,
    "inProgressAssignments" INTEGER NOT NULL DEFAULT 0,
    "notStartedAssignments" INTEGER NOT NULL DEFAULT 0,
    "averageScore" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "totalQuestions" INTEGER NOT NULL DEFAULT 0,
    "totalAnswers" INTEGER NOT NULL DEFAULT 0,
    "totalCorrectAnswers" INTEGER NOT NULL DEFAULT 0,
    "accuracyRate" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "completionRate" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "lastActivityDate" TIMESTAMP(3),
    "lastUpdated" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "student_stats_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."teacher_stats" (
    "id" TEXT NOT NULL,
    "teacherId" TEXT NOT NULL,
    "totalAssignments" INTEGER NOT NULL DEFAULT 0,
    "totalClasses" INTEGER NOT NULL DEFAULT 0,
    "totalStudents" INTEGER NOT NULL DEFAULT 0,
    "averageClassCompletion" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "averageClassScore" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "totalQuestions" INTEGER NOT NULL DEFAULT 0,
    "activeAssignments" INTEGER NOT NULL DEFAULT 0,
    "scheduledAssignments" INTEGER NOT NULL DEFAULT 0,
    "lastUpdated" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "teacher_stats_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."class_stats_detailed" (
    "id" TEXT NOT NULL,
    "classId" TEXT NOT NULL,
    "totalStudents" INTEGER NOT NULL DEFAULT 0,
    "totalAssignments" INTEGER NOT NULL DEFAULT 0,
    "averageCompletion" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "averageScore" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "totalQuestions" INTEGER NOT NULL DEFAULT 0,
    "totalAnswers" INTEGER NOT NULL DEFAULT 0,
    "totalCorrectAnswers" INTEGER NOT NULL DEFAULT 0,
    "accuracyRate" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "activeStudents" INTEGER NOT NULL DEFAULT 0,
    "studentsNeedingHelp" INTEGER NOT NULL DEFAULT 0,
    "lastActivityDate" TIMESTAMP(3),
    "lastUpdated" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "class_stats_detailed_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."school_stats" (
    "id" TEXT NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "totalUsers" INTEGER NOT NULL DEFAULT 0,
    "totalTeachers" INTEGER NOT NULL DEFAULT 0,
    "totalStudents" INTEGER NOT NULL DEFAULT 0,
    "totalClasses" INTEGER NOT NULL DEFAULT 0,
    "totalAssignments" INTEGER NOT NULL DEFAULT 0,
    "activeAssignments" INTEGER NOT NULL DEFAULT 0,
    "scheduledAssignments" INTEGER NOT NULL DEFAULT 0,
    "completedAssignments" INTEGER NOT NULL DEFAULT 0,
    "averageCompletionRate" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "averageScore" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "totalQuestions" INTEGER NOT NULL DEFAULT 0,
    "totalAnswers" INTEGER NOT NULL DEFAULT 0,
    "totalCorrectAnswers" INTEGER NOT NULL DEFAULT 0,
    "studentsNeedingHelp" INTEGER NOT NULL DEFAULT 0,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "completedStudents" INTEGER NOT NULL DEFAULT 0,
    "inProgressStudents" INTEGER NOT NULL DEFAULT 0,
    "notStartedStudents" INTEGER NOT NULL DEFAULT 0,

    CONSTRAINT "school_stats_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."performance_metrics" (
    "id" TEXT NOT NULL,
    "metricType" "public"."PerformanceMetricType" NOT NULL,
    "entityType" "public"."EntityType" NOT NULL,
    "entityId" TEXT NOT NULL,
    "timeFrame" "public"."TimeFrame" NOT NULL,
    "date" TIMESTAMP(3) NOT NULL,
    "hour" INTEGER,
    "value" DOUBLE PRECISION NOT NULL,
    "additionalData" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "performance_metrics_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."dashboard_snapshots" (
    "id" TEXT NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL,
    "snapshotType" "public"."DashboardSnapshotType" NOT NULL,
    "totalClasses" INTEGER,
    "totalTeachers" INTEGER,
    "totalStudents" INTEGER,
    "totalAssignments" INTEGER,
    "classAssignments" INTEGER,
    "individualAssignments" INTEGER,
    "averageCompletionRate" INTEGER,
    "averageSuccessRate" INTEGER,
    "studentsNeedingAttention" INTEGER,
    "recentActivities" INTEGER,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "publishedAt" TIMESTAMP(3),

    CONSTRAINT "dashboard_snapshots_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."upload_files" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "alternativeText" TEXT,
    "caption" TEXT,
    "width" INTEGER,
    "height" INTEGER,
    "formats" JSONB,
    "hash" TEXT NOT NULL,
    "ext" TEXT,
    "mime" TEXT NOT NULL,
    "size" DOUBLE PRECISION NOT NULL,
    "url" TEXT NOT NULL,
    "previewUrl" TEXT,
    "provider" TEXT NOT NULL,
    "providerMetadata" JSONB,
    "folderPath" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "folderId" TEXT,

    CONSTRAINT "upload_files_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."upload_folders" (
    "id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "pathId" INTEGER NOT NULL,
    "path" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "parentId" TEXT,

    CONSTRAINT "upload_folders_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."students_needing_help" (
    "id" TEXT NOT NULL,
    "studentId" TEXT NOT NULL,
    "reasons" JSONB NOT NULL,
    "needsHelpSince" TIMESTAMP(3) NOT NULL,
    "lastUpdated" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "isResolved" BOOLEAN NOT NULL DEFAULT false,
    "resolvedAt" TIMESTAMP(3),
    "overdueAssignments" INTEGER NOT NULL DEFAULT 0,
    "averageScore" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "completionRate" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    "daysNeedingHelp" INTEGER NOT NULL DEFAULT 0,
    "severity" TEXT NOT NULL DEFAULT 'MODERATE',
    "teacherNotes" TEXT,
    "actionsTaken" JSONB,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "students_needing_help_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "public"."students_needing_help_classes" (
    "studentNeedingHelpId" TEXT NOT NULL,
    "classId" TEXT NOT NULL,

    CONSTRAINT "students_needing_help_classes_pkey" PRIMARY KEY ("studentNeedingHelpId","classId")
);

-- CreateTable
CREATE TABLE "public"."students_needing_help_teachers" (
    "studentNeedingHelpId" TEXT NOT NULL,
    "teacherId" TEXT NOT NULL,

    CONSTRAINT "students_needing_help_teachers_pkey" PRIMARY KEY ("studentNeedingHelpId","teacherId")
);

-- CreateIndex
CREATE UNIQUE INDEX "users_username_key" ON "public"."users"("username");

-- CreateIndex
CREATE UNIQUE INDEX "users_email_key" ON "public"."users"("email");

-- CreateIndex
CREATE UNIQUE INDEX "roles_name_key" ON "public"."roles"("name");

-- CreateIndex
CREATE UNIQUE INDEX "evaluation_settings_assignmentId_key" ON "public"."evaluation_settings"("assignmentId");

-- CreateIndex
CREATE UNIQUE INDEX "quiz_submissions_quizId_studentId_sessionNumber_key" ON "public"."quiz_submissions"("quizId", "studentId", "sessionNumber");

-- CreateIndex
CREATE UNIQUE INDEX "quiz_answers_submissionId_questionId_key" ON "public"."quiz_answers"("submissionId", "questionId");

-- CreateIndex
CREATE UNIQUE INDEX "quiz_live_student_progress_sessionId_studentId_key" ON "public"."quiz_live_student_progress"("sessionId", "studentId");

-- CreateIndex
CREATE UNIQUE INDEX "student_sprites_studentId_key" ON "public"."student_sprites"("studentId");

-- CreateIndex
CREATE UNIQUE INDEX "stats_classes_classId_key" ON "public"."stats_classes"("classId");

-- CreateIndex
CREATE UNIQUE INDEX "assignment_stats_assignmentId_key" ON "public"."assignment_stats"("assignmentId");

-- CreateIndex
CREATE UNIQUE INDEX "student_stats_studentId_key" ON "public"."student_stats"("studentId");

-- CreateIndex
CREATE UNIQUE INDEX "teacher_stats_teacherId_key" ON "public"."teacher_stats"("teacherId");

-- CreateIndex
CREATE UNIQUE INDEX "class_stats_detailed_classId_key" ON "public"."class_stats_detailed"("classId");

-- CreateIndex
CREATE UNIQUE INDEX "school_stats_date_key" ON "public"."school_stats"("date");

-- CreateIndex
CREATE INDEX "performance_metrics_entityType_entityId_timeFrame_date_idx" ON "public"."performance_metrics"("entityType", "entityId", "timeFrame", "date");

-- CreateIndex
CREATE UNIQUE INDEX "students_needing_help_studentId_key" ON "public"."students_needing_help"("studentId");

-- AddForeignKey
ALTER TABLE "public"."users" ADD CONSTRAINT "users_roleId_fkey" FOREIGN KEY ("roleId") REFERENCES "public"."roles"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."role_permissions" ADD CONSTRAINT "role_permissions_permissionId_fkey" FOREIGN KEY ("permissionId") REFERENCES "public"."permissions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."role_permissions" ADD CONSTRAINT "role_permissions_roleId_fkey" FOREIGN KEY ("roleId") REFERENCES "public"."roles"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."user_classes" ADD CONSTRAINT "user_classes_classId_fkey" FOREIGN KEY ("classId") REFERENCES "public"."classes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."user_classes" ADD CONSTRAINT "user_classes_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."assignments" ADD CONSTRAINT "assignments_languageId_fkey" FOREIGN KEY ("languageId") REFERENCES "public"."languages"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."assignments" ADD CONSTRAINT "assignments_teacherId_fkey" FOREIGN KEY ("teacherId") REFERENCES "public"."users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."class_assignments" ADD CONSTRAINT "class_assignments_assignmentId_fkey" FOREIGN KEY ("assignmentId") REFERENCES "public"."assignments"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."class_assignments" ADD CONSTRAINT "class_assignments_classId_fkey" FOREIGN KEY ("classId") REFERENCES "public"."classes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."user_assignments" ADD CONSTRAINT "user_assignments_assignmentId_fkey" FOREIGN KEY ("assignmentId") REFERENCES "public"."assignments"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."user_assignments" ADD CONSTRAINT "user_assignments_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."evaluation_settings" ADD CONSTRAINT "evaluation_settings_assignmentId_fkey" FOREIGN KEY ("assignmentId") REFERENCES "public"."assignments"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."questions" ADD CONSTRAINT "questions_assignmentId_fkey" FOREIGN KEY ("assignmentId") REFERENCES "public"."assignments"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quizzes" ADD CONSTRAINT "quizzes_teacherId_fkey" FOREIGN KEY ("teacherId") REFERENCES "public"."users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_questions" ADD CONSTRAINT "quiz_questions_quizId_fkey" FOREIGN KEY ("quizId") REFERENCES "public"."quizzes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_options" ADD CONSTRAINT "quiz_options_questionId_fkey" FOREIGN KEY ("questionId") REFERENCES "public"."quiz_questions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_classes" ADD CONSTRAINT "quiz_classes_classId_fkey" FOREIGN KEY ("classId") REFERENCES "public"."classes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_classes" ADD CONSTRAINT "quiz_classes_quizId_fkey" FOREIGN KEY ("quizId") REFERENCES "public"."quizzes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_students" ADD CONSTRAINT "quiz_students_quizId_fkey" FOREIGN KEY ("quizId") REFERENCES "public"."quizzes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_students" ADD CONSTRAINT "quiz_students_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_submissions" ADD CONSTRAINT "quiz_submissions_quizId_fkey" FOREIGN KEY ("quizId") REFERENCES "public"."quizzes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_submissions" ADD CONSTRAINT "quiz_submissions_studentId_fkey" FOREIGN KEY ("studentId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_answers" ADD CONSTRAINT "quiz_answers_questionId_fkey" FOREIGN KEY ("questionId") REFERENCES "public"."quiz_questions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_answers" ADD CONSTRAINT "quiz_answers_submissionId_fkey" FOREIGN KEY ("submissionId") REFERENCES "public"."quiz_submissions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_live_sessions" ADD CONSTRAINT "quiz_live_sessions_quizId_fkey" FOREIGN KEY ("quizId") REFERENCES "public"."quizzes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_live_sessions" ADD CONSTRAINT "quiz_live_sessions_teacherId_fkey" FOREIGN KEY ("teacherId") REFERENCES "public"."users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_live_student_progress" ADD CONSTRAINT "quiz_live_student_progress_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "public"."quiz_live_sessions"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."quiz_live_student_progress" ADD CONSTRAINT "quiz_live_student_progress_studentId_fkey" FOREIGN KEY ("studentId") REFERENCES "public"."users"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."student_assignment_progress" ADD CONSTRAINT "student_assignment_progress_assignmentId_fkey" FOREIGN KEY ("assignmentId") REFERENCES "public"."assignments"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."student_assignment_progress" ADD CONSTRAINT "student_assignment_progress_questionId_fkey" FOREIGN KEY ("questionId") REFERENCES "public"."questions"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."student_assignment_progress" ADD CONSTRAINT "student_assignment_progress_studentId_fkey" FOREIGN KEY ("studentId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."activity_logs" ADD CONSTRAINT "activity_logs_assignmentId_fkey" FOREIGN KEY ("assignmentId") REFERENCES "public"."assignments"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."activity_logs" ADD CONSTRAINT "activity_logs_classId_fkey" FOREIGN KEY ("classId") REFERENCES "public"."classes"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."activity_logs" ADD CONSTRAINT "activity_logs_quizId_fkey" FOREIGN KEY ("quizId") REFERENCES "public"."quizzes"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."activity_logs" ADD CONSTRAINT "activity_logs_userId_fkey" FOREIGN KEY ("userId") REFERENCES "public"."users"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."assignment_categories" ADD CONSTRAINT "assignment_categories_assignmentGroupId_fkey" FOREIGN KEY ("assignmentGroupId") REFERENCES "public"."assignment_groups"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."student_sprites" ADD CONSTRAINT "student_sprites_spriteSetId_fkey" FOREIGN KEY ("spriteSetId") REFERENCES "public"."sprite_sets"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."student_sprites" ADD CONSTRAINT "student_sprites_studentId_fkey" FOREIGN KEY ("studentId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."stats_classes" ADD CONSTRAINT "stats_classes_classId_fkey" FOREIGN KEY ("classId") REFERENCES "public"."classes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."assignment_stats" ADD CONSTRAINT "assignment_stats_assignmentId_fkey" FOREIGN KEY ("assignmentId") REFERENCES "public"."assignments"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."student_stats" ADD CONSTRAINT "student_stats_studentId_fkey" FOREIGN KEY ("studentId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."teacher_stats" ADD CONSTRAINT "teacher_stats_teacherId_fkey" FOREIGN KEY ("teacherId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."class_stats_detailed" ADD CONSTRAINT "class_stats_detailed_classId_fkey" FOREIGN KEY ("classId") REFERENCES "public"."classes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."upload_files" ADD CONSTRAINT "upload_files_folderId_fkey" FOREIGN KEY ("folderId") REFERENCES "public"."upload_folders"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."upload_folders" ADD CONSTRAINT "upload_folders_parentId_fkey" FOREIGN KEY ("parentId") REFERENCES "public"."upload_folders"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."students_needing_help" ADD CONSTRAINT "students_needing_help_studentId_fkey" FOREIGN KEY ("studentId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."students_needing_help_classes" ADD CONSTRAINT "students_needing_help_classes_classId_fkey" FOREIGN KEY ("classId") REFERENCES "public"."classes"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."students_needing_help_classes" ADD CONSTRAINT "students_needing_help_classes_studentNeedingHelpId_fkey" FOREIGN KEY ("studentNeedingHelpId") REFERENCES "public"."students_needing_help"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."students_needing_help_teachers" ADD CONSTRAINT "students_needing_help_teachers_studentNeedingHelpId_fkey" FOREIGN KEY ("studentNeedingHelpId") REFERENCES "public"."students_needing_help"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "public"."students_needing_help_teachers" ADD CONSTRAINT "students_needing_help_teachers_teacherId_fkey" FOREIGN KEY ("teacherId") REFERENCES "public"."users"("id") ON DELETE CASCADE ON UPDATE CASCADE;

